using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Serialization;

// 功能：清空体素信息、渲染场景体素
public class VoxelGetIndirectLightRenderFeature : ScriptableRendererFeature
{
    public RenderTexture ilRT;
    class VoxelRenderPass : ScriptableRenderPass
    {
        public RenderTexture ilRT;
        
        // 临时RT
        /*static string rt_name = "_IndirectLight";
        private static int rt_id = Shader.PropertyToID(rt_name);*/
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            /*RenderTextureDescriptor descriptor = new RenderTextureDescriptor(VoxelGI.instance.voxTexSize, VoxelGI.instance.voxTexSize, RenderTextureFormat.Default, 0);
            cmd.GetTemporaryRT(rt_id, descriptor);*/
            ConfigureTarget(ilRT);
            ConfigureClear(ClearFlag.All, Color.black);
        }
        
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("GetIndirectLight");
            
            context.SetupCameraProperties(renderingData.cameraData.camera);
            cmd.SetRenderTarget(ilRT);
            context.ExecuteCommandBuffer(cmd);
            
            // 临时DrawSetting
            DrawingSettings drawSetting = new DrawingSettings(new ShaderTagId("IndirectLight"),
                new SortingSettings(Camera.main));
            FilteringSettings fs = new FilteringSettings(RenderQueueRange.all);
            ScriptableCullingParameters cullingParameters;
            renderingData.cameraData.camera.TryGetCullingParameters(false, out cullingParameters);
            CullingResults cullResults = context.Cull(ref cullingParameters);
            
            context.DrawRenderers(cullResults, ref drawSetting, ref fs);
            
            cmd.Clear();
            cmd.Release();
        }

        // Cleanup any allocated resources that were created during the execution of this render pass.
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
            //cmd.ReleaseTemporaryRT(rt_id);
        }
    }

    VoxelRenderPass m_ScriptablePass;

    /// <inheritdoc/>
    public override void Create()
    {
        m_ScriptablePass = new VoxelRenderPass();
        // Configures where the render pass should be injected.
        m_ScriptablePass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
        m_ScriptablePass.ilRT = ilRT;
    }
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(m_ScriptablePass);
    }
}


