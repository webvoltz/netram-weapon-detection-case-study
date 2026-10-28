// Excerpt from static/index.html ("// ..." marks omitted code)

const maxVideoFrameWidth = 1280;

// Output panel trails the video so each box is drawn on its own frame
const outputDelaySeconds = 1.0;
const captureFps = 12;
const maxInFlight = 2;
const maxQueueEntries = 45;
const maxBufferedBytes = 256 * 1024;
const logIntervalMs = 500;
const videoJpegQuality = 0.85;

// ...

async function startSyncedVideoSource(message) {
  activeKind = "video";
  videoMode = true;
  resetVideoPipeline();

  await waitForVideoReady();
  syncCanvasSize();
  drawOverlay();
  setStatus("Connecting backend...");
  await connectSocket();
  if (!videoMode) return;

  setStatus("Starting video...");
  await sourceVideo.play();
  if (!videoMode) return;

  running = true;
  webcamButton.disabled = true;
  videoFile.disabled = true;
  imageFile.disabled = true;
  stopButton.disabled = false;
  setStatus(message);

  requestAnimationFrame(captureVideoLoop);
  requestAnimationFrame(renderVideoLoop);
}

// ...

function captureVideoLoop(now) {
  if (!running || !videoMode) return;

  const dueForCapture = now - lastCaptureAt >= 1000 / captureFps;
  const socketReady = socket && socket.readyState === WebSocket.OPEN
    && socket.bufferedAmount < maxBufferedBytes;
  // Skip, never queue, when the backend or the link is behind
  if (dueForCapture && socketReady && inFlight < maxInFlight
      && !sourceVideo.ended && sourceVideo.videoWidth) {
    lastCaptureAt = now;
    captureAndSendVideoFrame();
  }

  requestAnimationFrame(captureVideoLoop);
}

function captureAndSendVideoFrame() {
  const scale = Math.min(1, maxVideoFrameWidth / sourceVideo.videoWidth);
  captureCanvas.width = Math.max(1, Math.round(sourceVideo.videoWidth * scale));
  captureCanvas.height = Math.max(1, Math.round(sourceVideo.videoHeight * scale));
  captureCtx.drawImage(sourceVideo, 0, 0, captureCanvas.width, captureCanvas.height);
  const mediaTime = sourceVideo.currentTime;

  inFlight++;
  const encodeStartedAt = performance.now();

  createImageBitmap(captureCanvas).then((bitmap) => {
    captureCanvas.toBlob((blob) => {
      if (!blob || !videoMode || !socket || socket.readyState !== WebSocket.OPEN) {
        bitmap.close();
        inFlight = Math.max(0, inFlight - 1);
        return;
      }
      pendingCaptures.push({
        time: mediaTime,
        bitmap,
        sentAt: performance.now(),
        encodeMs: performance.now() - encodeStartedAt,
        bytes: blob.size,
        bufferedBefore: socket.bufferedAmount
      });
      socket.send(blob);
    }, "image/jpeg", videoJpegQuality);
  }).catch(() => {
    inFlight = Math.max(0, inFlight - 1);
  });
}

function handleVideoResult(payload) {
  const receivedAt = performance.now();
  inFlight = Math.max(0, inFlight - 1);
  // Replies arrive in send order, so the oldest capture is this frame
  const capture = pendingCaptures.shift();
  if (!capture) return;

  if (payload.error) {
    capture.bitmap.close();
    setStatus(`Frame error: ${payload.error}`);
    return;
  }

  lastPayload = payload;
  renderQueue.push({ time: capture.time, bitmap: capture.bitmap, payload });
  while (renderQueue.length > maxQueueEntries) {
    renderQueue.shift().bitmap.close();
  }

  updateMetrics(payload);
  if (receivedAt - lastLoggedAt >= logIntervalMs) {
    lastLoggedAt = receivedAt;
    logPayload(payload, {
      roundTripMs: receivedAt - capture.sentAt,
      encodeMs: capture.encodeMs,
      bytes: capture.bytes,
      bufferedBefore: capture.bufferedBefore
    });
  }
}

function renderVideoLoop(now) {
  if (!running || !videoMode) return;

  if (sourceVideo.videoWidth) {
    syncCanvas(overlayCanvas, sourceVideo.videoWidth, sourceVideo.videoHeight);
    overlayCtx.drawImage(sourceVideo, 0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  const frameDelta = lastRenderNow ? (now - lastRenderNow) / 1000 : 0;
  lastRenderNow = now;

  const socketAlive = socket && socket.readyState === WebSocket.OPEN;
  const stillAnalysing = pendingCaptures.length > 0 || inFlight > 0;

  if (!sourceVideo.ended) {
    outputClock = sourceVideo.currentTime - outputDelaySeconds;
  } else if (renderQueue.length || !(stillAnalysing && socketAlive)) {
    outputClock += frameDelta;
  }

  while (renderQueue.length && renderQueue[0].time <= outputClock) {
    if (currentOutputEntry) currentOutputEntry.bitmap.close();
    currentOutputEntry = renderQueue.shift();
  }

  if (currentOutputEntry) {
    const { bitmap, payload } = currentOutputEntry;
    syncCanvas(outputCanvas, bitmap.width, bitmap.height);
    outputCtx.drawImage(bitmap, 0, 0, outputCanvas.width, outputCanvas.height);
    drawPayloadBoxes(payload, false);
  }

  const drained = !renderQueue.length && (!socketAlive || !stillAnalysing);
  if (sourceVideo.ended) {
    if (drained) {
      stop();
      setStatus("Video finished.");
      return;
    }
    setStatus(`Video ended - finishing analysis (${renderQueue.length + pendingCaptures.length} frame(s) left)...`);
  }

  requestAnimationFrame(renderVideoLoop);
}
