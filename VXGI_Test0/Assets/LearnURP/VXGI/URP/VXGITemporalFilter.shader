Shader "LearnURP/VXGITemporalFilter"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _LastFrameTex("Last Frame Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { 
            "RenderPipeline" = "UniversalPipeline"
        }
        // No culling or depth
        Cull Off ZWrite Off ZTest Always
        
        HLSLINCLUDE

        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
        //除了贴图外，要暴露在Inspector面板上的变量都需要缓存到CBUFFER中
        CBUFFER_START(UnityPerMaterial)
        CBUFFER_END
        
        ENDHLSL

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            struct Attributes
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varings
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            Varings vert (Attributes IN)
            {
                Varings OUT;
                OUT.vertex = TransformObjectToHClip(IN.vertex);
                OUT.uv = IN.uv;
                return OUT;
            }

            TEXTURE2D(_MainTex); SAMPLER(sampler_MainTex);
            TEXTURE2D(_LastFrameTex); SAMPLER(sampler_LastFrameTex);
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareNormalsTexture.hlsl"

            float4 frag (Varings IN) : SV_Target
            {
                half4 col = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, IN.uv);
                half4 lastCol = SAMPLE_TEXTURE2D(_LastFrameTex, sampler_LastFrameTex, IN.uv);
                if (length(col - lastCol) > 0.005) 
                    col = 0.1 * col + 0.9 * lastCol;
                col.w = 1;
                return col;
            }
            ENDHLSL
        }
    }
}
