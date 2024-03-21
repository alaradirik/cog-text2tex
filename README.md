## Cog Wrapper for Text2Tex

Text2Tex creates detailed textures for 3D meshes based on provided textual prompts. This approach integrates inpainting with a pre-trained 
depth-aware controlnet model, allowing for the gradual creation of high-resolution textures from various viewpoints.

See the [paper](https://daveredrum.github.io/Text2Tex/static/Text2Tex.pdf), [project page](https://daveredrum.github.io/Text2Tex/) and [original 
repository](https://github.com/daveredrum/Text2Tex) for more details.

## API Usage

You need to have Cog and Docker installed to run this model locally. To build the docker image with cog and run a prediction:

```
cog predict -i obj_file=@mesh.obj -i prompt="Orange backpack"
```

To start a server and send requests to your locally or remotely deployed API:

```
cog run -p 5000 python -m cog.server.http
```


To generate textures with Text2Tex, you need to provide a mesh file (.obj) and enter a text description of the texture you would like to generate. 
The API input arguments are as follows:

- **obj_file**: 3D object file (.obj) you would like to generate texture for. 
- **prompt**: text prompt to generate texture from.  
- **negative_prompt**: use this to specify what you don’t want in the texture, helping to refine the results.  
- **ddim_steps**: number of DDIM sampling steps, influencing the texture's progression and detail.  
- **new_strength**: percentage to determine the DDIM steps for generating new view.  
- **update_strength**: percentage to determine the DDIM steps to update the view.  
- **num_viewpoints**: number of different pre-determined viewpoints used to update the texture, more viewpoints result in consistent textures.  
- **viewpoint_mode**: strategy for selecting viewpoints, either predefined or hemispherical.  
- **update_steps**: number of iterations for texture updates, each enhancing quality or addressing specific texture artifacts.  
- **update_mode**: the method by which texture updates are applied through the iterations.  
- **seed**: seed for reproducibility, default value is None. Set to an arbitrary value for deterministic generation.  

## Usage Tips

Follow the recommended pre-processing steps for best quality texture generation:

- Y-axis is up.  
- The mesh should face towards +Z.  
- The mesh bounding box should be origin-aligned (note simply averaging the vertices coordinates could be problematic).  
- The max length of the mesh bounding box should be around 1.  

## References

```
@article{chen2023text2tex,
    title={Text2Tex: Text-driven Texture Synthesis via Diffusion Models},
    author={Chen, Dave Zhenyu and Siddiqui, Yawar and Lee, Hsin-Ying and Tulyakov, Sergey and Nie{\ss}ner, Matthias},
    journal={arXiv preprint arXiv:2303.11396},
    year={2023},
}
```
