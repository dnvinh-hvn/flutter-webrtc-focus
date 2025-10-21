package com.cloudwebrtc.webrtc.video

/**
 * Configuration for MediaPipe segmentation and auto-zoom feature
 */
data class SegmentationConfig(
    /** Enable/disable segmentation processing */
    val enabled: Boolean = true,
    
    /** Smoothing factor for crop transitions (0.0-1.0, higher = faster) */
    val smoothingFactor: Float = 0.15f,
    
    /** Padding factor around detected person (0.0-1.0) */
    val paddingFactor: Float = 0.2f,
    
    /** Process every Nth frame (higher = better performance, lower = smoother) */
    val frameSkip: Int = 3,
    
    /** Minimum confidence threshold for segmentation mask (0-255) */
    val segmentationThreshold: Int = 25
) {
    companion object {
        /** Default configuration for production */
        val DEFAULT = SegmentationConfig()
        
        /** High performance config for low-end devices */
        val PERFORMANCE = SegmentationConfig(
            smoothingFactor = 0.2f,
            frameSkip = 5
        )
        
        /** High quality config for high-end devices */
        val QUALITY = SegmentationConfig(
            smoothingFactor = 0.1f,
            frameSkip = 2
        )
        
        /** Disabled configuration */
        val DISABLED = SegmentationConfig(enabled = false)
    }
    
    init {
        require(smoothingFactor in 0.0f..1.0f) { "smoothingFactor must be between 0.0 and 1.0" }
        require(paddingFactor in 0.0f..1.0f) { "paddingFactor must be between 0.0 and 1.0" }
        require(frameSkip > 0) { "frameSkip must be positive" }
        require(segmentationThreshold in 0..255) { "segmentationThreshold must be between 0 and 255" }
    }
}