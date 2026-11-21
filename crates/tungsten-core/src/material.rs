/// Terrain materials. The discriminant is the stored sample value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Material {
    Air = 0,
    Stone = 1,
    Dirt = 2,
    Sand = 3,
}

pub const MATERIAL_COUNT: usize = 4;

/// How a material gives way under the mining brush.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CarveMode {
    /// Fill is worn away smoothly under the brush.
    Erode,
    /// Breaks off in world-anchored angular shards.
    Shards,
}

pub struct MaterialDef {
    pub name: &'static str,
    /// Linear RGB.
    pub color: [f32; 3],
    /// 0 meshes rounded and shades smooth, 1 meshes crisp and shades faceted.
    pub sharpness: f32,
    pub carve: CarveMode,
    /// Fill removed per tick at the brush centre, in 1/256 fill units.
    pub carve_rate: u32,
    /// Settles to its angle of repose after edits.
    pub slumps: bool,
}

/// Simulation ticks per second; carve rates are derived from it.
pub const TICK_HZ: u32 = 60;

/// Rate that takes a full sample to the surface threshold in `seconds`.
const fn rate_for(seconds_x100: u32) -> u32 {
    128 * 256 * 100 / (seconds_x100 * TICK_HZ)
}

pub const MATERIALS: [MaterialDef; MATERIAL_COUNT] = [
    MaterialDef {
        name: "air",
        color: [0.0, 0.0, 0.0],
        sharpness: 0.0,
        carve: CarveMode::Erode,
        carve_rate: 0,
        slumps: false,
    },
    MaterialDef {
        name: "stone",
        color: [0.17, 0.18, 0.20],
        sharpness: 1.0,
        carve: CarveMode::Shards,
        carve_rate: rate_for(120),
        slumps: false,
    },
    MaterialDef {
        name: "dirt",
        color: [0.135, 0.075, 0.036],
        sharpness: 0.0,
        carve: CarveMode::Erode,
        carve_rate: rate_for(40),
        slumps: false,
    },
    MaterialDef {
        name: "sand",
        color: [0.66, 0.53, 0.25],
        sharpness: 0.0,
        carve: CarveMode::Erode,
        carve_rate: rate_for(15),
        slumps: true,
    },
];

impl Material {
    pub const STONE: u8 = Material::Stone as u8;
    pub const DIRT: u8 = Material::Dirt as u8;
    pub const SAND: u8 = Material::Sand as u8;
    pub const AIR: u8 = Material::Air as u8;

    pub fn from_u8(v: u8) -> Material {
        match v {
            1 => Material::Stone,
            2 => Material::Dirt,
            3 => Material::Sand,
            _ => Material::Air,
        }
    }

    pub fn def(self) -> &'static MaterialDef {
        &MATERIALS[self as usize]
    }
}

#[inline]
pub fn def(material: u8) -> &'static MaterialDef {
    &MATERIALS[(material as usize).min(MATERIAL_COUNT - 1)]
}
