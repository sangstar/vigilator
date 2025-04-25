use pyo3::prelude::*;
use pyo3::{pyfunction, PyResult};
use tokio::net::TcpListener;
use axum::{
    extract::{Json, Path},
    http::{header, StatusCode},
    body::Body,
    routing::get,
    response::Response,
    Router,
};

use crate::db::{insert_output, query_field, TOKIO_RUNTIME};
use crate::server::handle_json;
use crate::outputs::{
    Field, ModelOutput, PyModelOutput, FieldName
};

// TODO: Will need to add some size limitation policy to the database so it doesn't
//       OOM from some many entries in memory

// TODO: Add HTTP server support to query the DB

#[pyfunction]
pub fn start_server() -> PyResult<()> {
    std::thread::spawn(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let app = Router::new().route("/", get(handle_json));
            let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();

            println!("Server running on http://0.0.0.0:3000");

            axum::serve(listener, app).await.unwrap();
        });
    });

    Ok(())
}

#[pyfunction]
pub fn send_output(text: String, token_ids: Vec<u32>, logits: Vec<f32>) -> PyResult<()> {
    let output = ModelOutput::new(text, token_ids, logits);
    TOKIO_RUNTIME.block_on(insert_output(&output)).unwrap();
    Ok(())
}

#[pyfunction]
pub fn retrieve_output_by_text(text: &str) -> PyResult<PyModelOutput> {
    let text_field = Field {
        field_name: FieldName::Text,
        value: text.to_string(),
    };
    let output: ModelOutput = TOKIO_RUNTIME.block_on(query_field(text_field))?;
    Ok(output.as_py_tuple())
}

#[pymodule]
fn vigilator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(send_output, m)?)?;
    m.add_function(wrap_pyfunction!(retrieve_output_by_text, m)?)?;
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    Ok(())
}
