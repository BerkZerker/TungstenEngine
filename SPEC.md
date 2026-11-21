# Tungsten Engine v0.1 Spec

This game engine is a procedurally generated, chunk-based, infinite world engine with chunks being streamed in and out from around the player, generated via multi-threading on the CPU and then rendered on the GPU. The world is not made up of blocks, or voxels, but is rather generated using noise for the terrain and then meshed into a polygon mesh. (For the initial builds the resolution of the mesh can be adjusted for a lower poly or smoother look). There will need to be some sort of data storage for the terrain though since it will be material based, and editable, such that different materials have different properties and will be deformed by mining in different ways. To start, we will have simple stone, dirt, and sand. Stone is jagged, dirt is smooth, and sand flattens out.

We only save edits to the disk, keeping the storage footprint low, so that the chunk can be re-generated on whatever client with just the seed + edits. Everything must be built for multiplayer to be implemented in the future from the groud up, so deterministic is important.

The initial version of this is just an infinite world, with those 3 materials, and a player char with WSAD, space for jump, shift for run, ctrl for crouch, left click for break, right click to place material, and a basic debugging view of text showing fps, position, etc. and what material. The number keys 1-3 will pick stone / dirt / sand.

This is going to be written in rust, from the ground up, and highly optimized. This initial version is on an m6 macbook pro, but we will want it to run on whatever GPU (nvidia, amd, intel, apple).
