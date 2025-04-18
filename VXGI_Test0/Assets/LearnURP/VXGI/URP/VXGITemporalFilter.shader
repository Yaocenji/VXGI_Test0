Shader "LearnURP/VXGITemporalFilter"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _LastFrameTex("Last Frame Texture", 2D) = "white" {}
        _MotionVector("Motion Vector", 2D) = "gray" {}
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
            TEXTURE2D(_MotionVector); SAMPLER(sampler_MotionVector);
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareNormalsTexture.hlsl"

            float4 frag (Varings IN) : SV_Target
            {
                float2 motionVector = SAMPLE_TEXTURE2D(_MotionVector, sampler_MotionVector, IN.uv).xy;
                
                half4 curCol = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, IN.uv);
                half4 lastCol = SAMPLE_TEXTURE2D(_LastFrameTex, sampler_LastFrameTex, IN.uv + motionVector);

                float3 color = 0;
                if (lastCol.x < 0 || lastCol.x > 1 || lastCol.y < 0 || lastCol.y > 1)
                    color = curCol;
                else
                {
                    //if (length(curCol - lastCol) > 0.005)
                        color = 0.1 * curCol + 0.9 * lastCol;
                }

                /*col.xy = motionVector * 0.5 + 0.5;
                col.z = 0;*/
                return float4(color, 1);
            }
            ENDHLSL
        }
    }
}
