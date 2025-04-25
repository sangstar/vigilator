use axum::{
    extract::Json,
    http::{header, StatusCode},
    body::Body,
    routing::get,
    response::Response,
    Router,
};
use axum::response::IntoResponse;
use serde_json::{Value, json};
use crate::db::query_field;
use crate::outputs::{Field, FieldName, ModelOutput};


#[tokio::main]
async fn main() {
    let app = Router::new().route("/", get(json));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

pub(crate) async fn json(Json(payload): Json<serde_json::Value>) -> Result<Response, StatusCode> {
    Ok(
        Json(json!({ "data": 42 })).into_response()
    )
}

pub async fn get_top_token_for_prompt(prompt: &str) -> (u32, f32) {
    let field = Field { field_name: FieldName::Text, value: prompt.to_string() };
    let output: ModelOutput = query_field(field).await.unwrap();
    output.get_top_token_id()
}
