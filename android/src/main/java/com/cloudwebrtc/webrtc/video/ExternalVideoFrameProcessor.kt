package com.cloudwebrtc.webrtc.video

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.webrtc.JavaI420Buffer
import org.webrtc.VideoFrame
import org.webrtc.YuvConverter
import org.webrtc.YuvHelper
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import androidx.core.graphics.scale

class ExternalVideoFrameProcessor(
    context: Context,
    private val outputWidth: Int,
    private val outputHeight: Int,
    private val enableSegmentation: Boolean = true,
    private val enableFaceDetection: Boolean = true
) : LocalVideoTrack.ExternalVideoFrameProcessing {
    
    private val TAG = "FrameProcessor"
    private val faceDetector: FaceDetectorWrapper? = FaceDetectorWrapper(context)

    private val lastCropRect = AtomicReference<RectF?>(null)
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val isProcessing = AtomicBoolean(false)
    private var frameCount = 0L
    private var detectedFrameCount = 0L
    private var croppedFrameCount = 0L
    private var SMOOTHING_FACTOR = 0.05f
    private var minWidthForUpdateCrop = 20f
    private val PADDING_FACTOR = 0.5f
    private val FRAME_SKIP = 20 // Process every 3rd
    private val LOG_INTERVAL = 30 // Log every 30 frames

    private var moveRectJob: Job? = null

    private var processorScope: CoroutineScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    override fun onFrame(frame: VideoFrame): VideoFrame {
        if (!enableSegmentation) {
            return frame
        }
        
        frameCount++

        if(lastCropRect.get() == null && frame.buffer.width > 1f && frame.buffer.height > 1f) {
            // Initialize with center crop
            val width = frame.buffer.width
            val height = frame.buffer.height
            val baseCropRect = RectF(
                width * 0.25f,
                height * 0.25f,
                width * 0.75f,
                height * 0.75f
            )
            lastCropRect.set(baseCropRect)
            minWidthForUpdateCrop = (frame.buffer.width.toFloat() / 512f) * 20f // Adjust smoothing based on resolution
        }

        var mustRelease = true
        // Process every Nth frame asynchronously to avoid blocking
        if (enableFaceDetection && frameCount % FRAME_SKIP == 0L && !isProcessing.get()) {
            mustRelease = false
            isProcessing.set(true)
            frame.retain() // Retain to keep alive during async processing

            processorScope.launch(Dispatchers.IO) {
                try {
                    processFrameAsync(frame)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in async processing", e)
                } finally {
                    try {
                        frame.release()
                    } catch (e: Exception) {
                        Log.e(TAG, "Error releasing frame: ${e.message}", e)
                    }
                    isProcessing.set(false)
                }
            }
        }
        
        // Always return original frame immediately (don't block)
        // The crop rect will be updated asynchronously and used for future frames
        return applyCropRect(frame, mustRelease)
    }
    
    private fun processFrameAsync(frame: VideoFrame) {
        val bitmap = videoFrameToBitmap(frame) ?: return

        try {
            val timestampMs = System.currentTimeMillis()
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = faceDetector?.processFrame(mpImage, timestampMs)
            val baseCropRect = RectF(
                bitmap.width * 0.25f,
                bitmap.height * 0.25f,
                bitmap.width * 0.75f,
                bitmap.height * 0.75f
            )
            var cropRect = baseCropRect
            result?.detections()?.firstOrNull()?.boundingBox()?.let { bbox ->
                detectedFrameCount++
                Log.d(TAG, "👤 Face detected with bbox: $bbox")
                // Compute crop rect with padding
                val faceCropRect = computeCropRectWithPadding(bbox, bitmap.width, bitmap.height)
                if(faceCropRect.width() > 0 && faceCropRect.height() > 0){
                    if(faceCropRect.width() < bitmap.width / 2f) {
                        val scaleW = (bitmap.width / 2f) - faceCropRect.width()
                        faceCropRect.left = (faceCropRect.left - scaleW / 2f).coerceAtLeast(0f)
                        faceCropRect.right = (faceCropRect.right + scaleW / 2f).coerceAtMost(bitmap.width.toFloat())
                    }
                    if(faceCropRect.height() < bitmap.height / 2f) {
                        val scaleH = (bitmap.height / 2f) - faceCropRect.height()
                        faceCropRect.top = (faceCropRect.top - scaleH / 2f).coerceAtLeast(0f)
                        faceCropRect.bottom = (faceCropRect.bottom + scaleH / 2f).coerceAtMost(bitmap.height.toFloat())
                    }
                    cropRect = faceCropRect
                }
            }
            smoothCropRect(cropRect)
        } catch (e: Exception) {
            Log.e(TAG, "Error in face detection processing", e)
        } finally {
            bitmap.recycle()
        }
    }
    private var previousBuffer: VideoFrame.Buffer? = null
    private fun applyCropRect(frame: VideoFrame, mustRelease: Boolean = true): VideoFrame {
        val cropRect = lastCropRect.get() ?: return frame
        
        val buffer = frame.buffer
        buffer.retain()

        val width = buffer.width
        val height = buffer.height
        val cropX = cropRect.left.toInt().coerceIn(0, width - 1)
        val cropY = cropRect.top.toInt().coerceIn(0, height - 1)
        val cropWidth = cropRect.width().toInt().coerceIn(1, width - cropX)
        val cropHeight = cropRect.height().toInt().coerceIn(1, height - cropY)

        // Skip cropping if rect is essentially the whole frame
        if (cropX < 5 && cropY < 5 && cropWidth > width - 10 && cropHeight > height - 10) {
            return frame
        }
        val i420Buffer = buffer.toI420()
        val scaled = JavaI420Buffer.cropAndScaleI420(i420Buffer, cropX, cropY, cropWidth, cropHeight, outputWidth, outputHeight)
        val croppedFrame = VideoFrame(scaled, frame.rotation, frame.timestampNs)
        i420Buffer?.release()
        buffer.release()
        previousBuffer?.release()
        previousBuffer = scaled
        return croppedFrame
    }
    
    private fun computeCropRectWithPadding(bbox: RectF, width: Int, height: Int): RectF {
        val bboxWidth = bbox.width()
        val bboxHeight = bbox.height()
        
        val paddingX = bboxWidth * PADDING_FACTOR
        val paddingY = bboxHeight * PADDING_FACTOR
        
        var left = bbox.left - paddingX
        var top = bbox.top - paddingY
        var right = bbox.right + paddingX
        var bottom = bbox.bottom + paddingY
        
        // Maintain aspect ratio
        val targetAspect = outputWidth.toFloat() / outputHeight
        val currentAspect = (right - left) / (bottom - top)
        
        if (currentAspect > targetAspect) {
            val newHeight = (right - left) / targetAspect
            val diff = newHeight - (bottom - top)
            top -= diff / 2
            bottom += diff / 2
        } else {
            val newWidth = (bottom - top) * targetAspect
            val diff = newWidth - (right - left)
            left -= diff / 2
            right += diff / 2
        }
        
        // Clamp to bounds
        left = left.coerceIn(0f, width.toFloat())
        top = top.coerceIn(0f, height.toFloat())
        right = right.coerceIn(0f, width.toFloat())
        bottom = bottom.coerceIn(0f, height.toFloat())
        
        return RectF(left, top, right, bottom)
    }
    
    private fun smoothCropRect(newRect: RectF) {
        moveRectJob?.cancel()
        moveRectJob = processorScope.launch(Dispatchers.IO) {
            val currentRect = lastCropRect.get() ?: run {
                lastCropRect.set(newRect)
                return@launch
            }

            var startRect = RectF(currentRect) // Make a copy
            if(abs(startRect.left - newRect.left) < minWidthForUpdateCrop) {
                return@launch
            }

            while (this.isActive && abs(startRect.left - newRect.left) >= SMOOTHING_FACTOR * 2f) {
                val interpolatedRect = RectF(
                    lerp(startRect.left, newRect.left, SMOOTHING_FACTOR),
                    lerp(startRect.top, newRect.top, SMOOTHING_FACTOR),
                    lerp(startRect.right, newRect.right, SMOOTHING_FACTOR),
                    lerp(startRect.bottom, newRect.bottom, SMOOTHING_FACTOR)
                )

                lastCropRect.set(interpolatedRect)
                startRect = RectF(interpolatedRect)
                kotlinx.coroutines.delay(30L) // Adjust delay for smoother/faster transitions
            }
            if(isActive) {
                lastCropRect.set(newRect) // Ensure final rect is exactly the target
            }
        }
    }

    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t

    /**
     * Convert a VideoFrame to a Bitmap for further processing.
     *
     * @param videoFrame The input VideoFrame to be converted.
     * @return The corresponding Bitmap representation of the VideoFrame.
     */
    private fun videoFrameToBitmap(videoFrame: VideoFrame): Bitmap? {
        try {
            // Convert the VideoFrame to I420 format
            val buffer = videoFrame.buffer
            val i420Buffer = buffer.toI420() ?: return null // Handle null case
            val y = i420Buffer.dataY
            val u = i420Buffer.dataU
            val v = i420Buffer.dataV
            val width = i420Buffer.width
            val height = i420Buffer.height
            val strides = intArrayOf(
                i420Buffer.strideY,
                i420Buffer.strideU,
                i420Buffer.strideV
            )

            // Convert I420 format to NV12 format as required by YuvImage
            val yuvBuffer = ByteBuffer.allocateDirect(width * height * 3 / 2)
            YuvHelper.I420ToNV12(
                y,
                strides[0],
                v,
                strides[2],
                u,
                strides[1],
                yuvBuffer,
                width,
                height
            )

            // Convert YuvImage to Bitmap
            val yuvImage = YuvImage(
                yuvBuffer.array(),
                ImageFormat.NV21,  // NV21 is compatible with NV12 for BitmapFactory
                width,
                height,
                null
            )
            val outputStream = ByteArrayOutputStream()
            yuvImage.compressToJpeg(
                Rect(0, 0, width, height),
                100,
                outputStream
            )
            val jpegData = outputStream.toByteArray()

            // Release resources
            i420Buffer.release()

            // Convert byte array to Bitmap
            return BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)
        } catch (e: Exception) {
            // Handle any exceptions and return null
            e.printStackTrace()
            return null
        }
    }
    
    fun close() {
        Log.i(TAG, "📊 Final Stats: Total frames=$frameCount, Detected=$detectedFrameCount, Cropped=$croppedFrameCount")
        Log.d(TAG, "Closing ExternalVideoFrameProcessor")
        moveRectJob?.cancel()
        processorScope.cancel()
        
        faceDetector?.close()
        executor.shutdown()
        try {
            if (!executor.awaitTermination(1, java.util.concurrent.TimeUnit.SECONDS)) {
                executor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            executor.shutdownNow()
        }
        previousBuffer?.release()
    }
}