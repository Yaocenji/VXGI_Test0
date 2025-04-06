using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

public class VisualizeVoxel : MonoBehaviour
{    
    [Header("体素分辨率")]
    public int voxTexSize = 256;
    [Header("体素大小")] 
    public float voxSize = 0.125f;
    
    private MeshCollider _mc;
    private MeshFilter _mf;
    private Mesh _mesh;
    Vector3[] _vertices;
    int[] _triangles;


    private int MAXNUM;
    // Start is called before the first frame update
    void Start()
    {
        
        _mc = GetComponent<MeshCollider>();
        _mf = GetComponent<MeshFilter>();

        MAXNUM = voxTexSize * voxTexSize * voxTexSize;
        _mesh = new Mesh();
        _mesh.indexFormat = IndexFormat.UInt32;
        generate();
        _mf.mesh = _mesh;
    }

    private void Update()
    {
    }

    private void generate()
    {
        _vertices = new Vector3[MAXNUM];
        
        Vector3 offset = new Vector3(-voxTexSize * voxSize / 2.0f,
                                     -voxTexSize * voxSize / 2.0f,
                                     -voxTexSize * voxSize / 2.0f);
        for (int i = 0; i < voxTexSize; i++) {
            for (int j = 0; j < voxTexSize; j++) {
                for (int k = 0; k < voxTexSize; k++) {
                    int idx = i * voxTexSize * voxTexSize + j * voxTexSize + k;
                    _vertices[idx] = new Vector3(i * voxSize, j * voxSize, k * voxSize);
                    _vertices[idx] += offset;
                }
            }
        }
        int triangleNum = ((MAXNUM - 1) / 3 + 1) * 3;
        _triangles = new int[triangleNum];
        for (int i = 0; i < triangleNum; i ++) {
            _triangles[i] = i;
            if (_triangles[i] >= MAXNUM) _triangles[i] -= MAXNUM;
        }

        _mesh.name = "voxelMesh";
        _mesh.vertices = _vertices;
        _mesh.triangles = _triangles;
        
        AssetDatabase.CreateAsset(_mesh, "Assets/voxelMesh.asset");
        AssetDatabase.SaveAssets();
    }

    /*private void OnDrawGizmos()
    {
        var s = new Vector3(voxSize, voxSize, voxSize) / 2.0f;
        if (_vertices == null) return;
        for (int i = 0; i < 30; i++)
        {
            for (int j = 0; j < 20; j++)
            {
                for (int k = 0; k < 6; k++)
                {
                    int idx = i * voxTexSize * voxTexSize + j * voxTexSize + k;
                    Gizmos.DrawCube(_vertices[idx], s);
                }
            }
        }
    }*/
}
