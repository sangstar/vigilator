use crate::db::insert_output;
use crate::outputs::ModelOutput;

pub mod outputs;
pub mod bindings;
pub mod db;



#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{query_field};

    #[tokio::test]
    async fn test_send_output() {
        pyo3::prepare_freethreaded_python();

        let token_ids = vec![1, 2, 3];
        let logits = vec![0.1, 0.2, 0.3];
        let text = "foo bar".to_string();

        let output = ModelOutput::new(text.clone(), token_ids.clone(), logits.clone());
        insert_output(&output).await.expect("Failed to insert output");

        let queried = query_field(output.text).await.expect("Failed querying data.");

        assert_eq!(queried.text.value, text);
        assert_eq!(queried.token_ids.value, token_ids);
        assert_eq!(queried.logits.value, logits);
    }
}
