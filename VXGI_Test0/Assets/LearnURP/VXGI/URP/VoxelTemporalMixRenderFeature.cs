using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class VoxelTemporalMixRenderFeature : ScriptableRendererFeature
{
    // 体素范围：以摄像机为中心的正方体
    // 设置范围大小
    [Header("体素信息计算着色器")]
    public ComputeShader manageVoxelDataCS;

    class VoxelRenderPass : ScriptableRenderPass
    {
        public VoxelGI _vxgiManager;
        
        public ComputeShader manageVoxelDataCS;
        private int temporalMixKernel;
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            // 在这里进行时间算法
            temporalMixKernel = manageVoxelDataCS.FindKernel("TemporalMix");
            manageVoxelDataCS.SetBuffer(temporalMixKernel, "VoxelTexture", VoxelGI.instance.voxelBuffer);
            manageVoxelDataCS.SetInt("voxTexSize", VoxelGI.instance.voxTexSize);
            //manageVoxelDataCS.Dispatch(temporalMixKernel, VoxelGI.instance.voxTexSize / 8, VoxelGI.instance.voxTexSize / 8, VoxelGI.instance.voxTexSize / 8);
        }
        
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            
            CommandBuffer cmd = CommandBufferPool.Get("temporalMixVoxelData");
            cmd.DispatchCompute(manageVoxelDataCS, temporalMixKernel, VoxelGI.instance.voxTexSize / 8, VoxelGI.instance.voxTexSize / 8, VoxelGI.instance.voxTexSize / 8);
            context.ExecuteCommandBuffer(cmd);

            cmd.Clear();
            cmd.Release();
            context.Submit();
        }

        // Cleanup any allocated resources that were created during the execution of this render pass.
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
        }
    }

    VoxelRenderPass m_ScriptablePass;

    /// <inheritdoc/>
    public override void Create()
    {
        m_ScriptablePass = new VoxelRenderPass();

        // Configures where the render pass should be injected.
        m_ScriptablePass.renderPassEvent = RenderPassEvent.BeforeRenderingOpaques;

        m_ScriptablePass.manageVoxelDataCS = manageVoxelDataCS;
    }
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(m_ScriptablePass);
    }
}


