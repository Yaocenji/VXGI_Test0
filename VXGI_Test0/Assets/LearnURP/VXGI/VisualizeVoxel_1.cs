using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class VisualizeVoxel_1 : MonoBehaviour
{
    private Transform camTrans;
    // Start is called before the first frame update
    void Start()
    {
        camTrans = Camera.main.transform;
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = camTrans.position;
    }
}
