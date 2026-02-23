#[derive(Debug)]
pub enum MetalRenderError {
    GpuDisabled,
    Timeout {
        stage: &'static str,
        timeout_ms: u64,
    },
    CommandBufferFailed {
        stage: &'static str,
    },
    OverflowDeferred {
        requested_capacity: usize,
        overlaps: u32,
    },
    Other(String),
}

impl std::fmt::Display for MetalRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuDisabled => f.write_str("Metal backend is disabled for this session"),
            Self::Timeout { stage, timeout_ms } => {
                write!(f, "Metal command timeout at {stage} after {timeout_ms}ms")
            }
            Self::CommandBufferFailed { stage } => {
                write!(f, "Metal command buffer failed at {stage}")
            }
            Self::OverflowDeferred {
                requested_capacity,
                overlaps,
            } => write!(
                f,
                "Metal overlap overflow deferred (requested_capacity={requested_capacity}, overlaps={overlaps})"
            ),
            Self::Other(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for MetalRenderError {}

impl MetalRenderError {
    pub fn should_disable_gpu(&self) -> bool {
        matches!(
            self,
            Self::GpuDisabled | Self::Timeout { .. } | Self::CommandBufferFailed { .. }
        )
    }
}

impl From<&str> for MetalRenderError {
    fn from(value: &str) -> Self {
        Self::Other(value.to_string())
    }
}

impl From<String> for MetalRenderError {
    fn from(value: String) -> Self {
        Self::Other(value)
    }
}

impl From<std::num::TryFromIntError> for MetalRenderError {
    fn from(err: std::num::TryFromIntError) -> Self {
        Self::Other(err.to_string())
    }
}
