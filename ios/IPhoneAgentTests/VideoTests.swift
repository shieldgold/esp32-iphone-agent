import XCTest
@testable import IPhoneAgent

/// `video` chunk reassembly (docs/ble-protocol.md).
final class VideoTests: XCTestCase {
    typealias F = VideoAssembler.Flags
    private static let decoderFrame = Data(base64Encoded: "AAAAAWdCwArcQmwEQAAAAwBAAAADAqPEieAAAAABaM4PyAAAAAFliIQ6EYoAAhjxwABA9jgACHlJycnXXXXXXXXXXXXg")!

    func chunk(_ seq: UInt16, _ flags: F, _ payload: [UInt8], ms: UInt32? = nil) -> Data {
        var d = Data([UInt8(seq & 0xff), UInt8(seq >> 8), flags.rawValue])
        if let ms { d.append(contentsOf: [UInt8(ms & 0xff), UInt8(ms >> 8 & 0xff), UInt8(ms >> 16 & 0xff), UInt8(ms >> 24)]) }
        d.append(contentsOf: payload)
        return d
    }

    let sps: [UInt8] = [0, 0, 0, 1, 0x67, 1, 2]
    let idr: [UInt8] = [0, 0, 0, 1, 0x65, 9, 9]
    let p: [UInt8] = [0, 0, 0, 1, 0x41, 7]

    func testInOrderChunksMakeAccessUnits() {
        var a = VideoAssembler()
        XCTAssertEqual(a.add(chunk(0, [.start, .end, .config], sps, ms: 1000))?.config, true)
        XCTAssertNil(a.add(chunk(1, [.start, .idr], Array(idr[0..<5]), ms: 1001)))
        let unit = a.add(chunk(2, [.end, .idr], Array(idr[5...])))
        XCTAssertEqual(unit, AccessUnit(data: Data(idr), idr: true, config: false, snap: false, boardMs: 1001))
        XCTAssertEqual(a.add(chunk(3, [.start, .end], p, ms: 1200))?.data, Data(p))
        XCTAssertEqual(a.drops, 0)
    }

    func testGapDropsUntilTheNextKeyUnit() {
        var a = VideoAssembler()
        _ = a.add(chunk(10, [.start, .end, .idr], idr, ms: 1))
        XCTAssertNil(a.add(chunk(11, [.start], p, ms: 2)))
        XCTAssertNil(a.add(chunk(13, [.end], p)))           // 12 lost: the half unit is dropped
        XCTAssertEqual(a.drops, 1)
        XCTAssertNil(a.add(chunk(14, [.start, .end], p, ms: 3)))  // a P-frame after a gap is useless
        XCTAssertNotNil(a.add(chunk(15, [.start, .end, .config], sps, ms: 4)))  // recovered on CONFIG
        XCTAssertNotNil(a.add(chunk(16, [.start, .end], p, ms: 5)))
    }

    func testRecoversOnIDRAndWrapsSeq() {
        var a = VideoAssembler()
        _ = a.add(chunk(65534, [.start, .end, .idr], idr, ms: 1))
        XCTAssertNotNil(a.add(chunk(65535, [.start, .end], p, ms: 2)))
        XCTAssertNotNil(a.add(chunk(0, [.start, .end], p, ms: 3)))  // u16 wrap is not a gap
        XCTAssertEqual(a.drops, 0)
        XCTAssertNil(a.add(chunk(5, [.start, .end], p, ms: 4)))
        XCTAssertEqual(a.drops, 4)
        let unit = a.add(chunk(6, [.start, .end, .idr, .snap], idr, ms: 5))
        XCTAssertEqual(unit?.snap, true)
    }

    func testWaitsForAKeyUnitAtStartAndIgnoresBench() {
        var a = VideoAssembler()
        XCTAssertNil(a.add(chunk(0, [.start, .end], p, ms: 1)))
        XCTAssertNil(a.add(chunk(1, [.bench], [UInt8](repeating: 0, count: 100))))
        XCTAssertEqual(a.benchBytes, 103)
        XCTAssertNotNil(a.add(chunk(2, [.start, .end, .idr], idr, ms: 2)))
    }

    func testAnnexBSplitsNALUnits() {
        let units = annexBUnits(Data(sps + [0, 0, 1, 0x68, 5] + idr))
        XCTAssertEqual(units.map { $0.first! & 0x1f }, [7, 8, 5])
        XCTAssertEqual(units[0], Data([0x67, 1, 2]))
    }

    func testDecoderCoalescesCatchUpFramesAndStillPublishesTheLastFrame() async {
        // A real 64x64 red H.264 IDR with SPS/PPS (libx264 baseline). Replaying a GOP must
        // decode its references without JPEG-encoding and publishing every historical frame.
        let data = Self.decoderFrame
        let decoder = H264Decoder()
        let latest = expectation(description: "last frame survives a burst without another input")
        let lock = NSLock()
        var frames: [UInt32] = []
        decoder.onFrame = { frame in
            lock.withLock { frames.append(frame.unit.boardMs) }
            if frame.unit.boardMs == 80 { latest.fulfill() }
        }
        decoder.onError = { XCTFail($0) }
        decoder.queue.async {
            for i in 1...80 {
                decoder.decode(AccessUnit(data: data, idr: true, config: false, snap: false, boardMs: UInt32(i)))
            }
        }
        await fulfillment(of: [latest], timeout: 3)
        let emitted = lock.withLock { frames }
        XCTAssertEqual(emitted.last, 80)
        XCTAssertLessThanOrEqual(emitted.count, 8, "Catch-up frames must not all become JPEGs/UI updates")
    }

    func testSnapshotFlushIncludesTheLastRateLimitedFrame() async {
        let decoder = H264Decoder()
        let lock = NSLock()
        var last: UInt32?
        decoder.onFrame = { frame in lock.withLock { last = frame.unit.boardMs } }
        decoder.onError = { XCTFail($0) }
        decoder.queue.async {
            decoder.decode(AccessUnit(data: Self.decoderFrame, idr: true, config: false, snap: true, boardMs: 1))
        }
        await decoder.flush()
        XCTAssertEqual(lock.withLock { last }, 1)
        decoder.queue.async {
            for i in 2...80 {
                decoder.decode(AccessUnit(data: Self.decoderFrame, idr: true, config: false, snap: true, boardMs: UInt32(i)))
            }
        }
        await decoder.flush()
        XCTAssertEqual(lock.withLock { last }, 80, "A snap must not return while its last frame is still pending")
    }

    @MainActor func testRecentlyReceivedHistoricalFrameIsNotTreatedAsLive() async throws {
        let board = FakeBoard()
        let device = CommandChannel(transport: board)
        board.channel = device
        device.linkUp()
        let feed = VideoFeed()
        feed.device = device
        feed.mirroring = true
        try await feed.stream(true)
        feed.boardClock(ms: 60_000)
        // show() publishes immediately, but its board timestamp is zero: the image is a minute old.
        feed.show(Data([1]), width: 64, height: 64)
        let source = try await feed.currentFrame(wait: 0)
        XCTAssertEqual(source, "snap")
        XCTAssertTrue(board.lines.contains { $0.hasSuffix(" snap") })
    }
}
