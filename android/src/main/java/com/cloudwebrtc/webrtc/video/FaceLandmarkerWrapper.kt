package com.cloudwebrtc.webrtc.video

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

class FaceLandmarkerWrapper(private val context: Context) {
    private var faceLandmarker: FaceLandmarker? = null
    private val TAG = "FaceLandmarker"
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
            baseOptionsBuilder.setModelAssetPath("face_landmarker.task")
            
            val options = FaceLandmarker.FaceLandmarkerOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumFaces(1)  // Optimize for single face (selfie)
                .setMinFaceDetectionConfidence(0.5f)
                .setMinFacePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setOutputFaceBlendshapes(false)
                .setOutputFacialTransformationMatrixes(false)
                .build()
            
            faceLandmarker = FaceLandmarker.createFromOptions(context, options)
            Log.d(TAG, "Face landmarker initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize face landmarker", e)
        }
    }
    
    /**
     * Process a frame and return face detection result
     */
    @Synchronized
    fun processFrame(mpImage: MPImage, timestampMs: Long): FaceLandmarkerResult? {
        return try {
            val landmarker = faceLandmarker ?: return null
            val result = landmarker.detectForVideo(mpImage, timestampMs)
            
            processedFrames++
            if (result != null && result.faceLandmarks().isNotEmpty()) {
                successfulDetections++
            }
            
            if (processedFrames % 30 == 0L) {
                val successRate = (successfulDetections * 100.0f / processedFrames).toInt()
                Log.d(TAG, "👤 Face Landmarker: Processed=$processedFrames, Detected=$successfulDetections (${successRate}%)")
            }
            
            result
        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame", e)
            null
        }
    }
    
    /**
     * Compute bounding box from face landmarks
     * Returns null if no face detected
     */
    fun computeBoundingBoxFromLandmarks(result: FaceLandmarkerResult?, width: Int, height: Int): RectF? {
        if (result == null || result.faceLandmarks().isEmpty()) {
            return null
        }
        
        try {
            val landmarksList = result.faceLandmarks()[0]  // First face
            
            // Find min/max normalized coordinates
            var minX = 1f
            var maxX = 0f
            var minY = 1f
            var maxY = 0f
            
            for (landmark in landmarksList.toList()) {
                val x = landmark.x()
                val y = landmark.y()
                
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
            
            // Convert normalized coordinates to pixels
            return RectF(
                minX * width,
                minY * height,
                maxX * width,
                maxY * height
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error computing bounding box", e)
            return null
        }
    }
    
    @Synchronized
    fun close() {
        Log.i(TAG, "👤 Face Landmarker Final: Processed=$processedFrames, Detected=$successfulDetections")
        faceLandmarker?.close()
        faceLandmarker = null
        Log.d(TAG, "Face landmarker closed")
    }
}