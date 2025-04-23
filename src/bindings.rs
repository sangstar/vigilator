use pyo3::{pyfunction, PyResult};
use pyo3::prelude::*;

use crate::db::{insert_output, query_output_from_id, TOKIO_RUNTIME};
use crate::outputs::ModelOutput;

// TODO: Include decoded text in the database as well so you can query by
//       a prompt

// TODO: Will need to add some size limitation policy to the database so it doesn't
//       OOM from some many entries in memory


#[pyfunction]
pub fn send_output(token_ids: Vec<u32>, logits: Vec<f32>) -> PyResult<()> {
    let output = ModelOutput::new(token_ids, logits);
    TOKIO_RUNTIME.block_on(insert_output(output)).unwrap();
    Ok(())
}


#[pyfunction]
pub fn retrieve_output(id: i32) -> PyResult<(Vec<u32>, Vec<f32>)> {
    let output: ModelOutput = TOKIO_RUNTIME.block_on(query_output_from_id(id))?;
    Ok((output.token_ids, output.logits))
}

#[pymodule]
fn vigilator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(send_output, m)?)?;
    m.add_function(wrap_pyfunction!(retrieve_output, m)?)?;
    Ok(())
}

