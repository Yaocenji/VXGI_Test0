using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.Serialization;

public class VoxelSpaceFilterRenderFeature : ScriptableRendererFeature
{
    [Header("空间滤波shader")]
    public Shader spaceFilterShader;

    [Header("间接光RT")] public RenderTexture ilRT;
    class SpaceFilterRenderPass : ScriptableRenderPass
    {
        public VoxelGI _vxgiManager;
        public Material spaceFilterMat;
        public RenderTexture ilRT;
        
        // 临时RT
        static string rt_name = "_VoxelGI_AfterSpaceFilter";
        private static int rt_id = Shader.PropertyToID(rt_name);
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            RenderTextureDescriptor descriptor = new RenderTextureDescriptor(1920, 1080, RenderTextureFormat.Default, 0);
            cmd.GetTemporaryRT(rt_id, descriptor);
            ConfigureTarget(rt_id);
            ConfigureClear(ClearFlag.Color, Color.white);
        }
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("SpaceFilter");
            
            /*cmd.Blit(renderingData.cameraData.renderer.cameraColorTarget, rt_id, spaceFilterMat);
            cmd.Blit(rt_id, renderingData.cameraData.renderer.cameraColorTarget);*/
            
            cmd.Blit(ilRT, rt_id, spaceFilterMat);
            cmd.Blit(rt_id, ilRT);
            
            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            cmd.Release();
            //context.Submit();
        }
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
            cmd.ReleaseTemporaryRT(rt_id);
        }
    }

    SpaceFilterRenderPass m_ScriptablePass;

    /// <inheritdoc/>
    public override void Create()
    {
        m_ScriptablePass = new SpaceFilterRenderPass();
        m_ScriptablePass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
        m_ScriptablePass.spaceFilterMat = new Material(spaceFilterShader);
        m_ScriptablePass.ilRT = ilRT;
    }
    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(m_ScriptablePass);
    }
}


