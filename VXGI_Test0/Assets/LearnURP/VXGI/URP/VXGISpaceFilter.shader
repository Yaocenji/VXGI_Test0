Shader "LearnURP/VXGISpaceFilter"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

            TEXTURE2D(_MainTex);
            SAMPLER(sampler_MainTex);
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareNormalsTexture.hlsl"

            float4 frag (Varings IN) : SV_Target
            {
                half4 col = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, IN.uv);
                // just invert the colors
                float3 normal = normalize(SAMPLE_TEXTURE2D(_CameraNormalsTexture, sampler_CameraNormalsTexture, IN.uv).xyz);
                float depth = SAMPLE_TEXTURE2D(_CameraDepthTexture, sampler_CameraDepthTexture, IN.uv).x;
                float depthEye = LinearEyeDepth(depth, _ZBufferParams);

                // 先实现高斯
                float3 ansCol = 0;
                float sigma = 3;
                float weight, sumWeight = 0;
                float dist2;
                for (int i = -sigma; i <= sigma; ++i)
                {
                    for (int j = -sigma; j <= sigma; ++j)
                    {
                        // 计算基础权重
                        dist2 = i * i + j * j;
                        weight = 0.5 / PI / sigma * exp(- dist2 * 0.5 / (sigma * sigma));
                        sumWeight += weight;

                        float2 currUV = IN.uv;
                        currUV += float2(i, j) * float2(_ScreenParams.z - 1, _ScreenParams.w - 1);

                        // 采样偏移数据
                        float4 currCol = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, currUV);
                        float3 currNormal = normalize(SAMPLE_TEXTURE2D(_CameraNormalsTexture, sampler_CameraNormalsTexture, currUV).xyz);
                        float currDepthEye = LinearEyeDepth(SAMPLE_TEXTURE2D(_CameraDepthTexture, sampler_CameraDepthTexture, currUV).x, _ZBufferParams);

                        // 计算法线、深度权重
                        float normalWeight = saturate(1.0 - length(currNormal - normal) / 1.414);
                        float depthWeight = exp(-abs(currDepthEye - depthEye) * sigma);
                        weight *= normalWeight * depthWeight;
                        
                        ansCol += currCol * weight;
                    }
                }
                ansCol /= sumWeight;
                
                col.xyz = ansCol;
                col.w = 1;
                
                return col;
            }
            ENDHLSL
        }
    }
}
