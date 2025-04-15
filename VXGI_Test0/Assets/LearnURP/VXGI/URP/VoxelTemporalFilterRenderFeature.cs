using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class VoxelTemporalFilterRenderFeature : ScriptableRendererFeature
{
    [Header("上一帧的RT")]
    public RenderTexture lastRT;
    
    [Header("时间滤波shader")]
    public Shader temporalFilterShader;
    [Header("组合shader")]
    public Shader combineLightShader;
    
    [Header("间接光RT")] public RenderTexture ilRT;

    class VoxelTemporalFilterRenderPass : ScriptableRenderPass
    {
        public RenderTexture lastRT;
        public Material temporalFilterMaterial;
        public Material combineLightMaterial;
        public RenderTexture ilRT;
        private static bool isFirstFrame = true;
        
        static string rt_name = "_VoxelGI_BeforeTemporalFilter";
        private static int rt_id = Shader.PropertyToID(rt_name);
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            RenderTextureDescriptor descriptor = new RenderTextureDescriptor(1920, 1080, RenderTextureFormat.Default, 0);
            cmd.GetTemporaryRT(rt_id, descriptor);
            ConfigureTarget(rt_id);
            ConfigureClear(ClearFlag.Color, Color.black);
            temporalFilterMaterial.SetTexture("_LastFrameTex", lastRT);
            combineLightMaterial.SetTexture("_IndirectLightTex", ilRT);
        }
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("TemporalFilter");
            
            /*cmd.Blit(renderingData.cameraData.renderer.cameraColorTarget, rt_id);
            cmd.Blit(rt_id, renderingData.cameraData.renderer.cameraColorTarget, temporalFilterMaterial);
            cmd.Blit(renderingData.cameraData.renderer.cameraColorTarget, lastRT);*/
            
            cmd.Blit(ilRT, rt_id);
            cmd.Blit(rt_id, ilRT, temporalFilterMaterial);
            cmd.Blit(ilRT, lastRT);
            
            cmd.Blit(renderingData.cameraData.renderer.cameraColorTarget, rt_id);
            cmd.Blit(rt_id, renderingData.cameraData.renderer.cameraColorTarget, combineLightMaterial);
            
            // 只看间接光
            // cmd.Blit(ilRT, renderingData.cameraData.renderer.cameraColorTarget);
            
            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            cmd.Release();
            context.Submit();
        }
        public override void OnCameraCleanup(CommandBuffer cmd)
        {
        }
    }

    VoxelTemporalFilterRenderPass thePass;

    /// <inheritdoc/>
    public override void Create()
    {
        lastRT.enableRandomWrite = true;
        Shader.SetGlobalTexture("_LastFrameTex", lastRT);
        
        thePass = new VoxelTemporalFilterRenderPass();
        thePass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
        thePass.lastRT = lastRT;
        thePass.temporalFilterMaterial = new Material(temporalFilterShader);
        thePass.combineLightMaterial = new Material(combineLightShader);
        thePass.ilRT = ilRT;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(thePass);
    }
}


