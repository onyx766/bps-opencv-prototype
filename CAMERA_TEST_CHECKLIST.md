# BPS Camera Test Checklist

Based on the 9/20 dual-cam test, the 9/21 camera test report, and the proposed Pi 5 architecture.

## Numbers already measured (don't re-test these)

- New AR0234 pair, 2080x1200 MJPEG at 60 FPS: **~209 KB/frame, ~12 MB/s per camera, ~24 MB/s for both** (from the 751 MB and 738 MB outputs for 3,600 frames in 62.4 s)
- Pi microSD write speed: **23.4 MB/s**. This is lower than the ~24 MB/s both cameras produce together, which explains the gaps and the 33.8 s flush delay
- Old ELP camera, 1920x1200 MJPEG: **~88.4 FPS** sustained over USB2
- Estimate for the old camera (not measured): about 17 MB/s per camera at 88 FPS. One fits on a USB2 bus. **Two do not fit on the same bus or hub**
- The 3,600 frames took 62.4 s, not 60 s. We can't tell whether that is startup time or dropped frames until per-frame timestamps are logged

---

## A. Before every test run (field tester)

- [ ] Plug cameras in by label and use `/dev/bps-left` and `/dev/bps-right` (serials 2607110001 / 2607110002). Never use `/dev/videoN`
- [ ] Put each camera on its **own USB port, not the shared USB-C hub**. Two cameras on one hub failed with `No space left on device`
- [ ] Save `lsusb -t` output. It shows each camera's link speed (5000M or 480M) and which bus it's on
- [ ] Save `v4l2-ctl -d <dev> --list-formats-ext` for each camera
- [ ] Set **manual exposure** below 11 ms for 90 FPS or below 16 ms for 60 FPS, and write down the exposure and gain values. Auto-exposure can quietly lower the FPS
- [ ] Record the output drive and its free space. Use NVMe or SSD, **never the microSD**
- [ ] Write down camera positions and height, lighting, and table size

## B. During the test

- [ ] Save the raw MJPEG stream with no re-encode (`ffmpeg -f v4l2 -input_format mjpeg ... -c copy`)
- [ ] Log a **timestamp for every frame**
- [ ] Run for **at least 10 minutes**. A 60-second run doesn't prove continuity
- [ ] Log CPU load and storage write speed (`top`/`htop`, `iostat -x 1`)
- [ ] Hit some real shots, including breaks and fast balls near the rails and pockets, so the footage is usable for scoring

## C. After each run

- [ ] Save `dmesg` output and look for USB resets, disconnects and `uvcvideo` errors
- [ ] Record per camera: frames requested vs recorded, duration, file size, **largest frame gap**, and the number of gaps over 2 frame intervals
- [ ] Keep each setup in **its own folder**: raw video, logs, notes, `lsusb`, `dmesg`
- [ ] Write down anything odd, such as a fallback to 480M, a camera disappearing, or a stutter in playback

## D. Test sequence (run in this order)

- [ ] **1. New AR0234 pair**, 2080x1200 @ 60, both at once, each on its own laptop port, 10+ min
- [ ] **2. Old ELP single**, 1920x1200 @ 90, 10+ min
- [ ] **3. Two old ELP cameras** (if a second one exists), 1920x1200 @ 90, **each on a separate USB controller**, 10+ min. This is the most important unanswered question
- [ ] **4. New AR0234**: check whether lower resolutions offer higher FPS, and whether 60 FPS still holds with a short manual exposure
- [ ] **5. Pi 5 + NVMe**: repeat run 1, and run 3 if possible, on the Pi with capture written to NVMe
- [ ] **6. Endurance**: the winning setup for 1+ hour on the Pi + NVMe

## E. Hardware and information still needed (client)

- [ ] Exact model number and lens/FOV of the old ELP camera (label, invoice, or `lsusb -v`)
- [ ] Buy storage: an **NVMe HAT on the Pi 5's PCIe port**, not a USB SSD, because a USB SSD shares USB bandwidth with the cameras
- [ ] Benchmark the NVMe write speed on the Pi before any camera test (`dd` or `fio`, 1+ GB)
- [ ] Check with ELP whether the **P02 trigger model** is available with a lens matching the old camera, plus its trigger voltage, polarity and pulse width
- [ ] Ask ELP why the USB3 pair only offers 60 FPS when the U3GS0234C is advertised at 120 FPS (firmware or board revision?)
- [ ] Retest Pi-to-laptop network speed with `iperf3`, and check the link speed with `ethtool eth0`. The 6 MB/s and 1.6 MB/s results look like a test setup problem, since Gigabit should give about 110 MB/s

## F. Software tasks (developer)

- [ ] Write a `bench_capture.py` script that opens 1 or 2 cameras by alias and logs per-frame timestamps, KB per frame, gaps and drops, writes raw MJPEG, and prints a summary
- [ ] Fix live capture in `main.py`: V4L2 backend, `MJPG` FOURCC set before the size, native resolution (1920x1200 / 2080x1200), requested FPS, manual exposure, and FPS measured from real frame timestamps instead of `CAP_PROP_FPS`
- [ ] Open cameras by `/dev/bps-left` / `/dev/bps-right` instead of by index
- [ ] Separate capture from processing: a capture thread feeds a ring buffer and the recorder, and detection pulls frames as fast as it can, so every frame is kept
- [ ] Profile detection FPS at full resolution on the laptop, then on the Pi 5. Decide between processing a reduced size, a cropped table region, or every Nth frame
- [ ] Dual-camera support (after the camera choice): two capture threads, frames paired by timestamp (or by trigger count if P02 is chosen), and a separate calibration per camera mapped into shared table coordinates

## G. Gates before locking hardware

- [ ] Two cameras stream at the target FPS at the same time on the real Pi USB layout
- [ ] Pi + NVMe records for 1+ hour with no gaps longer than 2 frames
- [ ] Sync is measured (or hardware-triggered) within the tolerance the tracking needs
- [ ] Lens/FOV covers the whole table without too much distortion
- [ ] Power (PoE++) and thermals hold over a long session
- [ ] The Pi, NVMe, cameras and sync board can each be replaced without taking the light bar apart
