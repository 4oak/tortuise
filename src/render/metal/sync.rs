use metal::MTLCommandBufferStatus;

use super::error::MetalRenderError;

pub fn commit_and_wait_with_timeout(
    command_buffer: &metal::CommandBufferRef,
    stage: &'static str,
    timeout: std::time::Duration,
) -> Result<(), MetalRenderError> {
    command_buffer.commit();
    let start = std::time::Instant::now();

    loop {
        match command_buffer.status() {
            MTLCommandBufferStatus::Completed => return Ok(()),
            MTLCommandBufferStatus::Error => {
                return Err(MetalRenderError::CommandBufferFailed { stage });
            }
            _ => {
                if start.elapsed() >= timeout {
                    return Err(MetalRenderError::Timeout {
                        stage,
                        timeout_ms: timeout.as_millis() as u64,
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

pub fn commit_and_wait_or_disable_gpu(
    command_buffer: &metal::CommandBufferRef,
    stage: &'static str,
    timeout: std::time::Duration,
    gpu_disabled: &mut bool,
) -> Result<(), MetalRenderError> {
    match commit_and_wait_with_timeout(command_buffer, stage, timeout) {
        Ok(()) => Ok(()),
        Err(err) => {
            if err.should_disable_gpu() {
                *gpu_disabled = true;
            }
            Err(err)
        }
    }
}
