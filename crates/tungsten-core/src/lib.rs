//! Deterministic core of the Tungsten engine.
//!
//! Everything that must be bit-identical on every machine lives here: world
//! generation, edit application, the sand simulation and the save codec. The
//! mesher and raycast live here too but are outside that contract.

pub mod chunk;
pub mod codec;
pub mod collide;
pub mod coords;
pub mod edit;
pub mod field;
pub mod hash;
pub mod material;
pub mod mesh;
pub mod noise;
pub mod raycast;
pub mod sand;
pub mod worldgen;

pub use chunk::{Chunk, Dense, Sample, Volume};
pub use coords::{CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, WorldParams};
pub use material::Material;
