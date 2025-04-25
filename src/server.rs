use axum::{
    extract::{Json, Path},
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
    let app = Router::new().route("/", get(handle_json));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

pub async fn remove_leading_and_trailing_quotes(text: String) -> String {
    let replaced = text.replace("\"", "");
    replaced
}

pub async fn handle_json(Json(payload): Json<serde_json::Value>) -> Result<Response, StatusCode> {

    if let Some(text) = payload.get("text") {
        if let Some(top_k) = payload.get("top_k") {
            let top_tokens = get_top_tokens_for_prompt(text.as_str().unwrap(), top_k.as_u64().unwrap() ).await;
            Ok(Json(json!({ "text": text, "top_k": top_tokens })).into_response())
        } else {
            let parsed_text = remove_leading_and_trailing_quotes(text.to_string()).await;
            let output: ModelOutput = query_field(Field { field_name: FieldName::Text, value: parsed_text}).await.unwrap();
            Ok(Json(json!(
                { "text": text, "logits": output.logits, "token_ids": output.token_ids }
            )).into_response())
        }
    } else {
        // Return an error if the expected key is missing
        Err(StatusCode::BAD_REQUEST)
    }
}

pub async fn get_top_tokens_for_prompt(prompt: &str, top_k: u64) -> Vec<(u32, f32)> {
    let field = Field { field_name: FieldName::Text, value: prompt.to_string() };
    let output: ModelOutput = query_field(field).await.unwrap();
    output.get_top_token_id(top_k)
}
