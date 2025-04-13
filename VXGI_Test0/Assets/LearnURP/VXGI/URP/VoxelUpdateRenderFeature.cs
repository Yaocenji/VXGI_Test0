using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Serialization;

// 功能：清空体素信息、渲染场景体素
public class VoxelUpdateRenderFeature : ScriptableRendererFeature
{
    // 体素范围：以摄像机为中心的正方体
    // 设置范围大小
    [Header("体素信息清除计算着色器")]
    public ComputeShader manageVoxelDataCS;

    class VoxelRenderPass : ScriptableRenderPass
    {
        public VoxelGI _vxgiManager;
        
        public ComputeShader manageVoxelDataCS;
        private int initKernel;
        
        // 临时RT
        static string rt_name = "_VoxelTextureTarget";
        private static int rt_id = Shader.PropertyToID(rt_name);
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            RenderTextureDescriptor descriptor = new RenderTextureDescriptor(VoxelGI.instance.voxTexSize, VoxelGI.instance.voxTexSize, RenderTextureFormat.Default, 0);
            cmd.GetTemporaryRT(rt_id, descriptor);
            ConfigureTarget(rt_id);
            ConfigureClear(ClearFlag.All, Color.black);
            
            // 初始化CS
            initKernel = manageVoxelDataCS.FindKernel("Clear");
            manageVoxelDataCS.SetBuffer(initKernel, "VoxelTexture", VoxelGI.instance.voxelBuffer);
            manageVoxelDataCS.SetInt("voxTexSize", VoxelGI.instance.voxTexSize);
            manageVoxelDataCS.Dispatch(initKernel, VoxelGI.instance.voxTexSize / 8, VoxelGI.instance.voxTexSize / 8, VoxelGI.instance.voxTexSize / 8);
        }
        
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            
            CommandBuffer cmd = CommandBufferPool.Get("updateVoxelData");
            cmd.DispatchCompute(manageVoxelDataCS, initKernel, VoxelGI.instance.voxTexSize / 8, VoxelGI.instance.voxTexSize / 8, VoxelGI.instance.voxTexSize / 8);
            context.ExecuteCommandBuffer(cmd);
            
            
            context.SetupCameraProperties(renderingData.cameraData.camera);
            cmd.SetRenderTarget(rt_id);
            context.ExecuteCommandBuffer(cmd);
            // 临时DrawSetting
            DrawingSettings drawSetting = new DrawingSettings(new ShaderTagId("Voxelization"),
                new SortingSettings(Camera.main));
            FilteringSettings fs = new FilteringSettings(RenderQueueRange.all);
            ScriptableCullingParameters cullingParameters;
            renderingData.cameraData.camera.TryGetCullingParameters(false, out cullingParameters);
            CullingResults cullResults = context.Cull(ref cullingParameters);
            
            context.DrawRenderers(cullResults, ref drawSetting, ref fs);
            
            
            
            cmd.Clear();
            cmd.Release();
            context.Submit();
        }

        // Cleanup any allocated resources that were created during the execution of this render pass.
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
            cmd.ReleaseTemporaryRT(rt_id);
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


