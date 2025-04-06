Shader "LearnURP/Lit"
{
    Properties
    {
        _BaseMap ("Base Texture",2D) = "white"{}
        _BaseColor("Base Color",Color)=(1,1,1,1)
        _SpecularColor("Specular Color",Color)=(1,1,1,1)
        _Smoothness("Smoothness", Float) = 0.5
    }
    SubShader
    {
        Tags { 
            "RenderPipeline" = "UniversalPipeline"
            "Queue"="Geometry"
            "RenderType"="Opaque" 
        }
        HLSLINCLUDE

        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
        //除了贴图外，要暴露在Inspector面板上的变量都需要缓存到CBUFFER中
        CBUFFER_START(UnityPerMaterial)
        float4 _BaseMap_ST;
        half4 _BaseColor;
        half4 _SpecularColor;
        half _Smoothness;
        CBUFFER_END
        
        ENDHLSL
        
        Pass
        {
            Tags{
                "LightMode" = "UniversalForward"
            }
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS : NORMAL;
                float2 uv : TEXCOORD;
            };

            struct Varings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 viewDirWS : TEXCOORD1;
                float3 normalWS : TEXCOORD2;
            };

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            Varings vert (Attributes IN)
            {
                Varings OUT;
                VertexPositionInputs positionInputs = GetVertexPositionInputs(IN.positionOS.xyz);
                VertexNormalInputs normalInputs = GetVertexNormalInputs(IN.normalOS.xyz);
                OUT.positionCS = positionInputs.positionCS;
                OUT.viewDirWS = GetCameraPositionWS() - positionInputs.positionWS;
                OUT.normalWS = normalInputs.normalWS;
                OUT.uv=TRANSFORM_TEX(IN.uv,_BaseMap);
                return OUT;
            }

            float4 frag (Varings IN) : SV_Target
            {
                // sample the texture
                half4 baseMap = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv);
                
                Light light = GetMainLight();
                float3 lightDirWS = light.direction;

                half3 diffuse = baseMap.xyz*_BaseColor*LightingLambert(light.color, light.direction, IN.normalWS);
                half3 specular = LightingSpecular(light.color, light.direction, normalize(IN.normalWS), normalize(IN.viewDirWS), _SpecularColor, _Smoothness);
                
                half3 color = diffuse + specular;
                return float4(color.xyz, 1);
            }
            ENDHLSL
        }
    }
}
