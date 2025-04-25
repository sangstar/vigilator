pub mod bindings;
pub mod db;
pub mod outputs;
mod server;

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::Json;
    use serde_json::json;
    use crate::db::query_field;
    use crate::server::get_top_token_for_prompt;
    use crate::db::insert_output;
    use crate::outputs::ModelOutput;


    #[tokio::test]
    async fn test_get_top_token_logprob() {
        pyo3::prepare_freethreaded_python();

        let token_ids = vec![1, 2, 3];
        let logits = vec![0.1, 0.2, 0.3];
        let text = "foo bar".to_string();

        let output = ModelOutput::new(text.clone(), token_ids.clone(), logits.clone());
        insert_output(&output)
            .await
            .expect("Failed to insert output");

        let results = get_top_token_for_prompt(&text).await;
        assert_eq!(results.1, 0.3)
    }

    #[tokio::test]
    async fn test_json_endpoint() {
        let payload = json!({ "key": "value" });

        let request = Request::builder()
            .method("POST")
            .uri("/")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap();

        let response = crate::server::json(Json(payload)).await.unwrap();

        // Assert the response status and body
        assert_eq!(response.status(), StatusCode::OK);
        dbg!(response.body());
    }

    #[tokio::test]
    async fn test_send_output() {
        pyo3::prepare_freethreaded_python();

        let token_ids = vec![1, 2, 3];
        let logits = vec![0.1, 0.2, 0.3];
        let text = "foo bar".to_string();

        let output = ModelOutput::new(text.clone(), token_ids.clone(), logits.clone());
        insert_output(&output)
            .await
            .expect("Failed to insert output");

        let queried = query_field(output.text)
            .await
            .expect("Failed querying data.");

        assert_eq!(queried.text.value, text);
        assert_eq!(queried.token_ids.value, token_ids);
        assert_eq!(queried.logits.value, logits);
    }
}
