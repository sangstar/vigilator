use pyo3::{pyfunction, PyResult};
use pyo3::prelude::*;

use crate::db::{insert_output, query_field, TOKIO_RUNTIME};
use crate::outputs::{ModelOutput, Field, PyModelOutput, DB_TEXT_FIELD_NAME,
                     DB_TOKEN_IDS_FIELD_NAME, DB_UUID_FIELD_NAME,
                     DB_TIME_FIELD_NAME, DB_LOGITS_FIELD_NAME};

// TODO: Include decoded text in the database as well so you can query by
//       a prompt

// TODO: Will need to add some size limitation policy to the database so it doesn't
//       OOM from some many entries in memory

// TODO: Add HTTP server support to query the DB

#[pyfunction]
pub fn send_output(text: String, token_ids: Vec<u32>, logits: Vec<f32>) -> PyResult<()> {
    let output = ModelOutput::new(text, token_ids, logits);
    TOKIO_RUNTIME.block_on(insert_output(&output)).unwrap();
    Ok(())
}


#[pyfunction]
pub fn retrieve_output_by_text(text: &str) -> PyResult<PyModelOutput> {
    let text_field = Field { field_name: DB_TEXT_FIELD_NAME.to_string(), value: text.to_string()};
    let output: ModelOutput = TOKIO_RUNTIME.block_on(query_field(text_field))?;
    Ok(output.as_py_tuple())
}

#[pymodule]
fn vigilator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(send_output, m)?)?;
    m.add_function(wrap_pyfunction!(retrieve_output_by_text, m)?)?;
    Ok(())
}

