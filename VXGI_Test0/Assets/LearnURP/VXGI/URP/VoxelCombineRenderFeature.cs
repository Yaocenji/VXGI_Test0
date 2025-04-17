using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class VoxelCombineRenderFeature : ScriptableRendererFeature
{
    [Header("组合shader")]
    public Shader combineLightShader;
    
    [Header("间接光RT")] public RenderTexture ilRT;

    class VoxelCombineRenderPass : ScriptableRenderPass
    {
        public Material combineLightMaterial;
        public RenderTexture ilRT;
        
        static string rt_name = "_BeforeCombine";
        private static int rt_id = Shader.PropertyToID(rt_name);
        
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            RenderTextureDescriptor descriptor = new RenderTextureDescriptor(1920, 1080, RenderTextureFormat.Default, 0);
            cmd.GetTemporaryRT(rt_id, descriptor);
            ConfigureTarget(rt_id);
            ConfigureClear(ClearFlag.Color, Color.black);
            combineLightMaterial.SetTexture("_IndirectLightTex", ilRT);
        }
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get("Combine Direct Light And Indirect Light");

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

    VoxelCombineRenderPass thePass;

    /// <inheritdoc/>
    public override void Create()
    {
        thePass = new VoxelCombineRenderPass();
        thePass.renderPassEvent = RenderPassEvent.AfterRenderingOpaques;
        thePass.combineLightMaterial = new Material(combineLightShader);
        thePass.ilRT = ilRT;
    }

    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(thePass);
    }
}


