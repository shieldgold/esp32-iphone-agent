import CoreMedia
import Foundation
import ImageIO
import UniformTypeIdentifiers
import VideoToolbox

/// Annex-B access units -> VideoToolbox -> JPEG (q 0.75). Runs on its own serial queue.
final class H264Decoder: @unchecked Sendable {  // confined to `queue`
    struct Frame {
        let jpeg: Data
        let width: Int, height: Int
        let decodeMs: Double
        let unit: AccessUnit
    }

    var onFrame: ((Frame) -> Void)?
    var onError: ((String) -> Void)?
    var quality: Double = 0.75
    let queue = DispatchQueue(label: "video.decode")

    private struct PendingFrame {
        let image: CVImageBuffer
        let decodeMs: Double
        let unit: AccessUnit
    }
    private let output = LatestFrameOutput<PendingFrame>(interval: 0.2)

    init() {
        output.consume = { [weak self] pending in
            guard let self, let frame = self.jpeg(pending.image, ms: pending.decodeMs, unit: pending.unit) else { return }
            self.onFrame?(frame)
        }
    }

    /// A snapshot must include the last decoded frame, even when its JPEG was waiting for the rate limit.
    func flush() async {
        await withCheckedContinuation { continuation in
            queue.async {
                if let session = self.session { VTDecompressionSessionWaitForAsynchronousFrames(session) }
                self.output.flush { continuation.resume() }
            }
        }
    }

    private var sps: Data?, pps: Data?
    private var format: CMVideoFormatDescription?
    private var session: VTDecompressionSession?
    /// Every unit since the last IDR. A still screen gets an IDR only every few minutes, so when iOS drops
    /// the session (e.g. going to background) the GOP is decoded again from here instead of fetched again
    /// over BLE (a snap of a long GOP takes up to a minute).
    private var gop: [AccessUnit] = []
    private var gopBytes = 0
    private static let gopMax = 64 << 20

    /// Decode one access unit (call on `queue`).
    func decode(_ unit: AccessUnit) {
        if unit.idr { (gop, gopBytes) = ([], 0) }
        if gopBytes + unit.data.count <= Self.gopMax, unit.idr || !gop.isEmpty {
            gop.append(unit)
            gopBytes += unit.data.count
        }
        let status = decodeOne(unit, emit: true)
        guard status != noErr else { return }
        resetSession()
        if gop.last?.data == unit.data, gop.first?.idr == true, replay() {
            let n = gop.count
            DispatchQueue.main.async { DebugLog.write("decoder: session lost (\(status)); decoded the GOP again (\(n) units)") }
            return
        }
        onError?("decode failed \(status)" + (status == kVTInvalidSessionErr ? " (session lost, e.g. in background)" : ""))
        resetSession()  // rebuilt at the next IDR/config
    }

    /// Decode the cached GOP into a fresh session; only its last frame is shown.
    private func replay() -> Bool {
        for (i, u) in gop.enumerated() where decodeOne(u, emit: i == gop.count - 1) != noErr { return false }
        return true
    }

    /// noErr also when there is nothing to decode yet (no slices, no SPS/PPS).
    private func decodeOne(_ unit: AccessUnit, emit: Bool) -> OSStatus {
        var slices: [Data] = []
        for nal in annexBUnits(unit.data) {
            switch nal[nal.startIndex] & 0x1f {
            case 7: if nal != sps { sps = nal; resetSession() }
            case 8: if nal != pps { pps = nal; resetSession() }
            case 9, 6: break  // access unit delimiter, SEI
            default: slices.append(nal)
            }
        }
        guard !slices.isEmpty else { return noErr }
        guard ensureSession(), let session, let format else { return noErr }
        var avcc = Data()
        for s in slices {
            var len = UInt32(s.count).bigEndian
            avcc.append(Data(bytes: &len, count: 4))
            avcc.append(s)
        }
        guard let sample = makeSample(avcc, format) else { return noErr }
        let started = CFAbsoluteTimeGetCurrent()
        var flags = VTDecodeInfoFlags()
        let status = VTDecompressionSessionDecodeFrame(session, sampleBuffer: sample, flags: [],
                                                       infoFlagsOut: &flags) { [weak self] status, _, image, _, _ in
            guard let self else { return }
            guard status == noErr, let image else {
                self.onError?("decode callback status \(status)")
                return
            }
            guard emit else { return }
            let ms = (CFAbsoluteTimeGetCurrent() - started) * 1000
            self.output.offer(PendingFrame(image: image, decodeMs: ms, unit: unit))
        }
        return status
    }

    private func resetSession() {
        if let session { VTDecompressionSessionInvalidate(session) }
        session = nil
        format = nil
    }

    private func ensureSession() -> Bool {
        if session != nil { return true }
        guard let sps, let pps else { return false }
        var desc: CMFormatDescription?
        let ok = sps.withUnsafeBytes { s in
            pps.withUnsafeBytes { p in
                let ptrs = [s.bindMemory(to: UInt8.self).baseAddress!, p.bindMemory(to: UInt8.self).baseAddress!]
                let sizes = [sps.count, pps.count]
                return CMVideoFormatDescriptionCreateFromH264ParameterSets(
                    allocator: nil, parameterSetCount: 2, parameterSetPointers: ptrs, parameterSetSizes: sizes,
                    nalUnitHeaderLength: 4, formatDescriptionOut: &desc)
            }
        }
        guard ok == noErr, let desc else { onError?("bad SPS/PPS \(ok)"); return false }
        let attrs: [CFString: Any] = [kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA]
        var made: VTDecompressionSession?
        let status = VTDecompressionSessionCreate(allocator: nil, formatDescription: desc, decoderSpecification: nil,
                                                  imageBufferAttributes: attrs as CFDictionary,
                                                  outputCallback: nil, decompressionSessionOut: &made)
        guard status == noErr, let made else { onError?("decoder create failed \(status)"); return false }
        format = desc
        session = made
        return true
    }

    private func makeSample(_ avcc: Data, _ format: CMVideoFormatDescription) -> CMSampleBuffer? {
        var block: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(allocator: nil, memoryBlock: nil, blockLength: avcc.count,
                                                 blockAllocator: nil, customBlockSource: nil, offsetToData: 0,
                                                 dataLength: avcc.count, flags: 0, blockBufferOut: &block) == noErr,
              let block else { return nil }
        let copied = avcc.withUnsafeBytes {
            CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: block, offsetIntoDestination: 0,
                                          dataLength: avcc.count)
        }
        guard copied == noErr else { return nil }
        var sample: CMSampleBuffer?
        var size = avcc.count
        CMSampleBufferCreateReady(allocator: nil, dataBuffer: block, formatDescription: format, sampleCount: 1,
                                  sampleTimingEntryCount: 0, sampleTimingArray: nil, sampleSizeEntryCount: 1,
                                  sampleSizeArray: &size, sampleBufferOut: &sample)
        return sample
    }

    private func jpeg(_ buffer: CVImageBuffer, ms: Double, unit: AccessUnit) -> Frame? {
        var image: CGImage?
        VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &image)
        guard let image else { onError?("frame -> image failed"); return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.jpeg.identifier as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return Frame(jpeg: out as Data, width: image.width, height: image.height, decodeMs: ms, unit: unit)
    }
}

/// Decode every H.264 reference, but encode/publish at most five JPEGs per second. A catch-up burst
/// keeps only its latest pixel buffer; the trailing frame is delivered even if the screen then stops.
private final class LatestFrameOutput<Value>: @unchecked Sendable {
    var consume: (Value) -> Void = { _ in }
    private let interval: TimeInterval
    private let queue = DispatchQueue(label: "video.jpeg")
    private let lock = NSLock()
    private var pending: Value?
    private var scheduled = false
    private var lastOutput = -Double.infinity  // confined to queue

    init(interval: TimeInterval) { self.interval = interval }

    func offer(_ value: Value) {
        let start = lock.withLock {
            pending = value
            guard !scheduled else { return false }
            scheduled = true
            return true
        }
        if start { queue.async { self.work() } }
    }

    func flush(_ completion: @escaping () -> Void) {
        queue.async {
            self.deliver(forced: true)
            completion()
        }
    }

    private func work() {
        let delay = lastOutput + interval - ProcessInfo.processInfo.systemUptime
        if delay > 0 {
            queue.asyncAfter(deadline: .now() + delay) { self.work() }
        } else { deliver(forced: false) }
    }

    private func deliver(forced: Bool) {
        let value = lock.withLock {
            if !forced { scheduled = false }
            defer { pending = nil }
            return pending
        }
        guard let value else { return }
        consume(value)
        lastOutput = ProcessInfo.processInfo.systemUptime
    }
}
