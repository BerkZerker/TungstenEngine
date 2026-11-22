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
    /// Cut away smoothly to the shape of the brush.
    Smooth,
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
    /// How fast the brush sinks into it while mining, metres per second.
    pub dig_speed: f32,
}

/// Simulation ticks per second.
pub const TICK_HZ: u32 = 60;

pub const MATERIALS: [MaterialDef; MATERIAL_COUNT] = [
    MaterialDef { name: "air", color: [0.0, 0.0, 0.0], sharpness: 0.0, carve: CarveMode::Smooth, dig_speed: 0.0 },
    MaterialDef { name: "stone", color: [0.17, 0.18, 0.20], sharpness: 1.0, carve: CarveMode::Shards, dig_speed: 1.0 },
    MaterialDef {
        name: "dirt",
        color: [0.135, 0.075, 0.036],
        sharpness: 0.0,
        carve: CarveMode::Smooth,
        dig_speed: 2.5,
    },
    MaterialDef { name: "sand", color: [0.66, 0.53, 0.25], sharpness: 0.0, carve: CarveMode::Smooth, dig_speed: 5.0 },
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
