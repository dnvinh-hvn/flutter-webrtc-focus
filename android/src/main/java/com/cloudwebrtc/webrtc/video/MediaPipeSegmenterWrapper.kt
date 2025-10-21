package com.cloudwebrtc.webrtc.video

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import java.nio.ByteBuffer

class MediaPipeSegmenterWrapper(private val context: Context, private val onResult: (ImageSegmenterResult?) -> Unit) {
    private var imageSegmenter: ImageSegmenter? = null
    private val TAG = "MediaPipeSegmenter"
    private var processedFrames = 0L
    private var successfulDetections = 0L
    
    init {
        try {
            val baseOptionsBuilder = BaseOptions.builder()
            
            // Try GPU first, fallback to CPU
            try {
                baseOptionsBuilder.setDelegate(Delegate.GPU)
                Log.d(TAG, "Using GPU delegate")
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate failed, falling back to CPU", e)
                baseOptionsBuilder.setDelegate(Delegate.CPU)
            }
            
            // Use the model from assets
            baseOptionsBuilder.setModelAssetPath("selfie_segmenter.tflite")
            
            val options = ImageSegmenter.ImageSegmenterOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setOutputCategoryMask(true)
                .setOutputConfidenceMasks(false)
                .setResultListener { result: ImageSegmenterResult, bitmap ->
                    processedFrames++
                    if (result != null && !result.categoryMask().isEmpty) {
                        successfulDetections++
                    }

                    if (processedFrames % 30 == 0L) {
                        val successRate = (successfulDetections * 100.0f / processedFrames).toInt()
                        Log.d(TAG, "🔬 MediaPipe: Processed=$processedFrames, Success=$successfulDetections (${successRate}%)")
                    }
                    onResult(result)
                }
                .build()
            
            imageSegmenter = ImageSegmenter.createFromOptions(context, options)
            Log.d(TAG, "MediaPipe segmenter initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize MediaPipe segmenter", e)
        }
    }
    
    /**
     * Process a frame and return segmentation mask
     */
    @Synchronized
    fun processFrame(mpImage: MPImage, timestampMs: Long) {
         try {
            val segmenter = imageSegmenter ?: return
            segmenter.segmentAsync(mpImage, timestampMs)
             mpImage.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame", e)
            null
        }
    }
    
    /**
     * Compute bounding box from segmentation mask
     * Returns null if no valid foreground detected
     */
    @RequiresApi(Build.VERSION_CODES.N)
    fun computeBoundingBoxFromMask(result: ImageSegmenterResult?, width: Int, height: Int): RectF? {
        if (result == null || result.categoryMask().isEmpty) {
            return null
        }
        
        try {
            val mask = result.categoryMask().get()
            val maskWidth = mask.width
            val maskHeight = mask.height
            val buffer = ByteBufferExtractor.extract(mask).asFloatBuffer()
            
            var minX = maskWidth
            var minY = maskHeight
            var maxX = 0
            var maxY = 0
            var hasPixels = false
            
            // Find bounding box of foreground pixels (value > 0)
            for (y in 0 until maskHeight) {
                for (x in 0 until maskWidth) {
                    val index = y * maskWidth + x
                    val value = buffer.get(index).toInt() and 0xFF
                    
                    // Threshold for foreground (adjust as needed)
                    if (value > 25) { // ~10% threshold
                        hasPixels = true
                        if (x < minX) minX = x
                        if (x > maxX) maxX = x
                        if (y < minY) minY = y
                        if (y > maxY) maxY = y
                    }
                }
            }
            
            if (!hasPixels) {
                return null
            }
            
            // Convert mask coordinates to original image coordinates
            val scaleX = width.toFloat() / maskWidth
            val scaleY = height.toFloat() / maskHeight
            
            return RectF(
                minX * scaleX,
                minY * scaleY,
                maxX * scaleX,
                maxY * scaleY
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error computing bounding box", e)
            return null
        }
    }
    
    @Synchronized
    fun close() {
        Log.i(TAG, "🔬 MediaPipe Final: Processed=$processedFrames, Successful=$successfulDetections")
        imageSegmenter?.close()
        imageSegmenter = null
        Log.d(TAG, "MediaPipe segmenter closed")
    }
}