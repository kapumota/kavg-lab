//! Librería interna de KAvgLab.
//!
//! Los módulos se exponen para que los tests de integración puedan validar
//! configuraciones reales y ejecutar los flujos principales sin depender del binario.

pub mod attention;
pub mod config;
pub mod fenchel;
pub mod functions;
pub mod io;
pub mod kernels;
pub mod math;
pub mod optimization;
pub mod parallel;
pub mod profile;
pub mod prox;
pub mod suite;
