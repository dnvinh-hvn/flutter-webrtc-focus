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
import com.google.mediapipe.framework.image.MPImage
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

class ExternalVideoFrameProcessor(
    private val context: Context,
    private val outputWidth: Int,
    private val outputHeight: Int,
    private val enableSegmentation: Boolean = true
) : LocalVideoTrack.ExternalVideoFrameProcessing {
    
    private val TAG = "FrameProcessor"
    private val faceDetector: FaceDetectorWrapper? = FaceDetectorWrapper(context)

    private val lastCropRect = AtomicReference<RectF?>(null)
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val isProcessing = AtomicBoolean(false)
    private var frameCount = 0L
    private var detectedFrameCount = 0L
    private var croppedFrameCount = 0L
    private val SMOOTHING_FACTOR = 0.05f
    private val PADDING_FACTOR = 0.5f
    private val FRAME_SKIP = 2 // Process every 3rd frame for detection
    private val yuvConverter = YuvConverter()
    private val LOG_INTERVAL = 30 // Log every 30 frames
    
    override fun onFrame(frame: VideoFrame): VideoFrame {
        if (!enableSegmentation) {
            return frame
        }
        
        frameCount++
        
        // Process every Nth frame asynchronously to avoid blocking
        if (frameCount % FRAME_SKIP == 0L && !isProcessing.get()) {
            isProcessing.set(true)
            frame.retain() // Retain to keep alive during async processing
            
            executor.execute {
                try {
                    processFrameAsync(frame)
                } catch (e: Exception) {
                    Log.e(TAG, "Error in async processing", e)
                } finally {
                    frame.release()
                    isProcessing.set(false)
                }
            }
        }
        
        // Always return original frame immediately (don't block)
        // The crop rect will be updated asynchronously and used for future frames
        return applyCropRect(frame)
    }
    
    private fun processFrameAsync(frame: VideoFrame) {
        val bitmap = videoFrameToBitmap(frame,) ?: return

        try {
            val timestampMs = System.currentTimeMillis()
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = faceDetector?.processFrame(mpImage, timestampMs)
            Log.d(TAG, "Face detector processed in ${System.currentTimeMillis() - timestampMs} ms")
            Log.d(TAG, "Faces detected: ${result?.detections()?.size ?: 0}")
            val baseCropRect = RectF(
                bitmap.width * 0.25f,
                bitmap.height * 0.25f,
                bitmap.width * 0.75f,
                bitmap.height * 0.75f
            )
            var cropRect = baseCropRect
            result?.detections()?.firstOrNull()?.boundingBox()?.let { bbox ->
                detectedFrameCount++

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
    
    private fun applyCropRect(frame: VideoFrame): VideoFrame {
        val cropRect = lastCropRect.get() ?: return frame
        
        val buffer = frame.buffer

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
        
        try {
            val bitmap = videoFrameToBitmap(frame) ?: return frame
            val croppedBitmap = Bitmap.createBitmap(bitmap, cropX, cropY, cropWidth, cropHeight)
            val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, outputWidth, outputHeight, true)
            croppedFrameCount++
            if (croppedFrameCount % LOG_INTERVAL == 1L) {
                Log.i(TAG, "✂️ Cropped frame #$croppedFrameCount: crop=[$cropX,$cropY,${cropWidth}x$cropHeight] → output=${outputWidth}x$outputHeight")
                val runtime = Runtime.getRuntime()
                Log.d(TAG, "Memory: ${(runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024}MB used")
            }
            val croppedFrame = convertBitmapToVideoFrame(scaledBitmap) ?: return frame
            scaledBitmap.recycle()
            croppedBitmap.recycle()
            return croppedFrame
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "⚠️ Out of memory in applyCropRect! Clearing crop to recover")
            lastCropRect.set(null)
            return frame
        } catch (e: Exception) {
            Log.e(TAG, "Error applying crop rect: ${e.message}", e)
            return frame
        }
    }
    private fun convertBitmapToVideoFrame(bitmap: Bitmap): VideoFrame? {
        // Create the buffer for the video frame
        val width = bitmap.width
        val height = bitmap.height

        // Calculate the size of the Y, U, and V buffers
        val ySize = width * height
        val uvSize = ySize / 4

        // Allocate buffers for Y, U, and V planes
        val yBuffer = ByteBuffer.allocateDirect(ySize)
        val uBuffer = ByteBuffer.allocateDirect(uvSize)
        val vBuffer = ByteBuffer.allocateDirect(uvSize)

        // Lock the bitmap to get the pixel data
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Fill the Y, U, and V buffers with the pixel data
        for (i in pixels.indices) {
            val color = pixels[i]

            // Extract the R, G, and B components
            val r = (color shr 16) and 0xFF
            val g = (color shr 8) and 0xFF
            val b = color and 0xFF

            // Calculate Y, U, and V values
            val y = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
            val u = (-0.169 * r - 0.331 * g + 0.5 * b + 128).toInt()
            val v = (0.5 * r - 0.419 * g - 0.081 * b + 128).toInt()

            // Fill the Y buffer
            yBuffer.put(y.toByte())

            // Fill the U and V buffers (4:2:0 subsampling)
            if (i % 2 == 0 && (i / width) % 2 == 0) {
                uBuffer.put(u.toByte())
                vBuffer.put(v.toByte())
            }
        }

        // Rewind the buffers to prepare for reading
        yBuffer.rewind()
        uBuffer.rewind()
        vBuffer.rewind()

        // Create the I420 buffer
        val i420Buffer = JavaI420Buffer.wrap(
            width, height,
            yBuffer, width,
            uBuffer, width / 2,
            vBuffer, width / 2,
            null
        )

        // Create the video frame
        return VideoFrame(i420Buffer, 0, System.nanoTime())
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
    
    private fun smoothCropRect(newRect: RectF): RectF {
        val prev = lastCropRect.get()
        if (prev == null) {
            lastCropRect.set(newRect)
            return newRect
        }
        
        val smoothed = RectF(
            lerp(prev.left, newRect.left, SMOOTHING_FACTOR),
            lerp(prev.top, newRect.top, SMOOTHING_FACTOR),
            lerp(prev.right, newRect.right, SMOOTHING_FACTOR),
            lerp(prev.bottom, newRect.bottom, SMOOTHING_FACTOR)
        )
        
        lastCropRect.set(smoothed)
        return smoothed
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
        
        faceDetector?.close()
        yuvConverter.release()
        executor.shutdown()
        try {
            if (!executor.awaitTermination(1, java.util.concurrent.TimeUnit.SECONDS)) {
                executor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            executor.shutdownNow()
        }
    }
}