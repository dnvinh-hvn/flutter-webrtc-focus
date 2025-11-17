package com.cloudwebrtc.webrtc.video

import android.util.Log
//import com.google.mediapipe.framework.image.MPImage
//import com.google.mediapipe.tasks.core.BaseOptions
//import com.google.mediapipe.tasks.core.Delegate
//import com.google.mediapipe.tasks.vision.core.RunningMode
//import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
//import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult

class FaceDetectorWrapper(private val context: android.content.Context) {

    val TAG = "FaceDetector"

//    lateinit var faceDetector: FaceDetector
//
//    init {
//        val baseOptionsBuilder = BaseOptions.builder()
//        try {
//            baseOptionsBuilder.setDelegate(Delegate.GPU)
//            Log.d(TAG, "Using GPU delegate")
//        } catch (e: Exception) {
//            Log.w(TAG, "GPU delegate failed, falling back to CPU", e)
//            baseOptionsBuilder.setDelegate(Delegate.CPU)
//        }
//        baseOptionsBuilder.setModelAssetPath("blaze_face_short_range.tflite")
//        val options =
//            FaceDetector.FaceDetectorOptions.builder()
//                .setBaseOptions(baseOptionsBuilder.build())
//                .setMinDetectionConfidence(0.5f)
//                .setMinSuppressionThreshold(0.5f)
//                .setRunningMode(RunningMode.VIDEO).build()
//
//        faceDetector = FaceDetector.createFromOptions(context, options)
//    }

    var processedFrames = 0L
    var successfulDetections = 0L

//    @Synchronized
//    fun processFrame(mpImage: MPImage, timestampMs: Long): FaceDetectorResult? {
//        return try {
//            val landmarker = faceDetector ?: return null
//            val result = landmarker.detectForVideo(mpImage, timestampMs)
//
//            processedFrames++
//            if (result != null && result.detections().isNotEmpty()) {
//                successfulDetections++
//            }
//
//            if (processedFrames % 30 == 0L) {
//                val successRate = (successfulDetections * 100.0f / processedFrames).toInt()
//                Log.d(TAG, "👤 Face Landmarker: Processed=$processedFrames, Detected=$successfulDetections (${successRate}%)")
//            }
//
//            result
//        } catch (e: Exception) {
//            Log.e(TAG, "Error processing frame", e)
//            null
//        }
//    }
//
//    fun close() {
//        faceDetector.close()
//    }
}