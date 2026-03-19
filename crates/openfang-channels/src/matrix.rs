//! Matrix channel adapter.
//!
//! Uses the Matrix Client-Server API (via reqwest) for sending and receiving messages.
//! Implements /sync long-polling for real-time message reception.

use crate::types::{ChannelAdapter, ChannelContent, ChannelMessage, ChannelType, ChannelUser, OutputFormat};
use crate::formatter;
use async_trait::async_trait;
use chrono::Utc;
use futures::Stream;
use openfang_runtime::llm_driver::StreamEvent;
use serde_json;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, watch, RwLock};
use tracing::{debug, info, warn};
use zeroize::Zeroizing;

const SYNC_TIMEOUT_MS: u64 = 30000;
const MAX_MESSAGE_LEN: usize = 4096;

// Streaming constants
const EDIT_INTERVAL_MS: u64 = 500;
const MIN_EDIT_CHARS: usize = 50;
const MAX_PREVIEW_LINES: usize = 3;
const MAX_INPUT_CHARS: usize = 128;
const MAX_OUTPUT_CHARS: usize = 256;

// Tool call formatting (Chinese labels)
const TOOL_CALLS_HEADER: &str = "🔧 工具调用";
const INPUT_LABEL: &str = "📥 输入";
const OUTPUT_LABEL: &str = "✅ 输出";
const ERROR_LABEL: &str = "❌ 错误";
const NO_PARAMS: &str = "(无参数)";
const INDENT_PREFIX: &str = "　　";

/// Matrix channel adapter using the Client-Server API.
pub struct MatrixAdapter {
    /// Matrix homeserver URL (e.g., `"https://matrix.org"`).
    homeserver_url: String,
    /// Bot's user ID (e.g., "@openfang:matrix.org").
    user_id: String,
    /// SECURITY: Access token is zeroized on drop.
    access_token: Zeroizing<String>,
    /// HTTP client.
    client: reqwest::Client,
    /// Allowed room IDs (empty = all joined rooms).
    allowed_rooms: Vec<String>,
    /// Shutdown signal.
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
    /// Sync token for resuming /sync.
    since_token: Arc<RwLock<Option<String>>>,
    /// Whether to auto-accept room invites.
    auto_accept_invites: bool,
}

impl MatrixAdapter {
    /// Create a new Matrix adapter.
    pub fn new(
        homeserver_url: String,
        user_id: String,
        access_token: String,
        allowed_rooms: Vec<String>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            homeserver_url,
            user_id,
            access_token: Zeroizing::new(access_token),
            client: reqwest::Client::new(),
            allowed_rooms,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
            since_token: Arc::new(RwLock::new(None)),
            auto_accept_invites: true,
        }
    }

    /// Send a text message to a Matrix room.
    /// Returns the event_id of the sent message.
    async fn api_send_message(
        &self,
        room_id: &str,
        text: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let chunks = crate::types::split_message(text, MAX_MESSAGE_LEN);
        let mut last_event_id = String::new();

        for chunk in chunks {
            // Each chunk needs a unique transaction ID
            let txn_id = uuid::Uuid::new_v4().to_string();
            let url = format!(
                "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
                self.homeserver_url, room_id, txn_id
            );

            let body = serde_json::json!({
                "msgtype": "m.text",
                "body": chunk,
            });

            let resp = self
                .client
                .put(&url)
                .bearer_auth(&*self.access_token)
                .json(&body)
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(format!("Matrix API error {status}: {body}").into());
            }

            // Parse response to get event_id
            let resp_body: serde_json::Value = resp.json().await?;
            last_event_id = resp_body["event_id"].as_str().unwrap_or("").to_string();
        }

        Ok(last_event_id)
    }

    /// Send an HTML-formatted message to a Matrix room.
    ///
    /// Matrix supports rich text via `org.matrix.custom.html` format.
    /// The `plain_text` fallback is shown on clients that don't support HTML.
    /// Returns the event_id of the sent message.
    async fn api_send_html_message(
        &self,
        room_id: &str,
        plain_text: &str,
        html: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        // Generate plain text fallback by stripping HTML tags
        let fallback = if plain_text.is_empty() {
            strip_html_tags(html)
        } else {
            plain_text.to_string()
        };

        // Check if the message fits in a single chunk
        let chunks = crate::types::split_message(&fallback, MAX_MESSAGE_LEN);
        let mut last_event_id = String::new();

        if chunks.len() == 1 && html.len() <= MAX_MESSAGE_LEN * 2 {
            // Single message with HTML formatting
            let txn_id = uuid::Uuid::new_v4().to_string();
            let url = format!(
                "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
                self.homeserver_url, room_id, txn_id
            );

            debug!(
                "Matrix: sending HTML message to room {}, html_len={}, plain_len={}",
                room_id,
                html.len(),
                fallback.len()
            );

            let body = serde_json::json!({
                "msgtype": "m.text",
                "body": fallback,
                "format": "org.matrix.custom.html",
                "formatted_body": html,
            });

            let resp = self
                .client
                .put(&url)
                .bearer_auth(&*self.access_token)
                .json(&body)
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(format!("Matrix API error {status}: {body}").into());
            }

            // Parse response to get event_id
            let resp_body: serde_json::Value = resp.json().await?;
            last_event_id = resp_body["event_id"].as_str().unwrap_or("").to_string();
        } else {
            // Message needs chunking - send as plain text without HTML
            // (splitting HTML would break the structure)
            warn!(
                "Matrix: message too long, sending as plain text ({} chunks, html_len={})",
                chunks.len(),
                html.len()
            );

            for chunk in chunks {
                // Each chunk needs a unique transaction ID
                let txn_id = uuid::Uuid::new_v4().to_string();
                let url = format!(
                    "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
                    self.homeserver_url, room_id, txn_id
                );

                let body = serde_json::json!({
                    "msgtype": "m.text",
                    "body": chunk,
                });

                let resp = self
                    .client
                    .put(&url)
                    .bearer_auth(&*self.access_token)
                    .json(&body)
                    .send()
                    .await?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(format!("Matrix API error {status}: {body}").into());
                }

                // Parse response to get event_id
                let resp_body: serde_json::Value = resp.json().await?;
                last_event_id = resp_body["event_id"].as_str().unwrap_or("").to_string();
            }
        }

        Ok(last_event_id)
    }

    /// Edit an existing message using Matrix's m.replace relation.
    ///
    /// Returns the new event ID on success.
    async fn api_edit_message(
        &self,
        room_id: &str,
        original_event_id: &str,
        new_content: &str,
        new_html: Option<&str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let txn_id = uuid::Uuid::new_v4().to_string();
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
            self.homeserver_url, room_id, txn_id
        );

        let plain = strip_html_tags(new_content);
        let body = if let Some(html) = new_html {
            serde_json::json!({
                "msgtype": "m.text",
                "body": format!("* {}", plain),
                "format": "org.matrix.custom.html",
                "formatted_body": html,
                "m.new_content": {
                    "msgtype": "m.text",
                    "body": plain,
                    "format": "org.matrix.custom.html",
                    "formatted_body": html
                },
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": original_event_id
                }
            })
        } else {
            serde_json::json!({
                "msgtype": "m.text",
                "body": format!("* {}", new_content),
                "m.new_content": {
                    "msgtype": "m.text",
                    "body": new_content
                },
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": original_event_id
                }
            })
        };

        let resp = self
            .client
            .put(&url)
            .bearer_auth(&*self.access_token)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let resp_body = resp.text().await.unwrap_or_default();
            return Err(format!("Matrix edit API error {status}: {resp_body}").into());
        }

        let resp_body: serde_json::Value = resp.json().await?;
        let event_id = resp_body["event_id"].as_str().unwrap_or("").to_string();
        Ok(event_id)
    }

    /// Validate credentials by calling /whoami.
    async fn validate(&self) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/_matrix/client/v3/account/whoami", self.homeserver_url);

        let resp = self
            .client
            .get(&url)
            .bearer_auth(&*self.access_token)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err("Matrix authentication failed".into());
        }

        let body: serde_json::Value = resp.json().await?;
        let user_id = body["user_id"].as_str().unwrap_or("unknown").to_string();

        Ok(user_id)
    }

    #[cfg(test)]
    fn is_allowed_room(&self, room_id: &str) -> bool {
        self.allowed_rooms.is_empty() || self.allowed_rooms.iter().any(|r| r == room_id)
    }
}

/// Accept a room invite by calling POST /_matrix/client/v3/rooms/{room_id}/join.
async fn accept_invite(
    client: &reqwest::Client,
    homeserver: &str,
    access_token: &str,
    room_id: &str,
) {
    let url = format!("{homeserver}/_matrix/client/v3/rooms/{room_id}/join");
    match client
        .post(&url)
        .bearer_auth(access_token)
        .json(&serde_json::json!({}))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            info!("Matrix: auto-accepted invite to {room_id}");
        }
        Ok(resp) => {
            let status = resp.status();
            warn!("Matrix: failed to accept invite to {room_id}: {status}");
        }
        Err(e) => {
            warn!("Matrix: error accepting invite to {room_id}: {e}");
        }
    }
}

/// Get the number of joined members in a room.
async fn get_room_member_count(
    client: &reqwest::Client,
    homeserver: &str,
    access_token: &str,
    room_id: &str,
) -> Option<usize> {
    let url = format!("{homeserver}/_matrix/client/v3/rooms/{room_id}/joined_members");
    let resp = client
        .get(&url)
        .bearer_auth(access_token)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    body["joined"].as_object().map(|m| m.len())
}

/// Do an initial /sync with timeout=0 to get the since token without processing events.
/// This prevents replaying old messages when the adapter first connects.
async fn initial_sync(
    client: &reqwest::Client,
    homeserver: &str,
    access_token: &str,
) -> Option<String> {
    let url = format!(
        "{homeserver}/_matrix/client/v3/sync?timeout=0&filter={{\"room\":{{\"timeline\":{{\"limit\":0}}}}}}"
    );
    let resp = client
        .get(&url)
        .bearer_auth(access_token)
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    body["next_batch"].as_str().map(String::from)
}

/// Convert a Matrix mxc:// URL to a download URL.
///
/// Matrix media URLs are in the format: mxc://serverName/mediaId
/// Download URL format: {homeserver}/_matrix/media/v3/download/{serverName}/{mediaId}
fn mxc_to_download_url(homeserver: &str, mxc_url: &str) -> Option<String> {
    if !mxc_url.starts_with("mxc://") {
        return None;
    }
    // Remove "mxc://" prefix
    let rest = &mxc_url[6..];
    // Split into serverName and mediaId
    let parts: Vec<&str> = rest.splitn(2, '/').collect();
    if parts.len() != 2 {
        return None;
    }
    let server_name = parts[0];
    let media_id = parts[1];
    
    // Build download URL
    // Remove trailing slash from homeserver if present
    let homeserver = homeserver.trim_end_matches('/');
    Some(format!(
        "{}/_matrix/media/v3/download/{}/{}",
        homeserver, server_name, media_id
    ))
}

#[async_trait]
impl ChannelAdapter for MatrixAdapter {
    fn name(&self) -> &str {
        "matrix"
    }

    fn channel_type(&self) -> ChannelType {
        ChannelType::Matrix
    }

    async fn start(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = ChannelMessage> + Send>>, Box<dyn std::error::Error>>
    {
        // Validate credentials
        let validated_user = self.validate().await?;
        info!("Matrix adapter authenticated as {validated_user}");

        let (tx, rx) = mpsc::channel::<ChannelMessage>(256);
        let homeserver = self.homeserver_url.clone();
        let access_token = self.access_token.clone();
        let user_id = self.user_id.clone();
        let allowed_rooms = self.allowed_rooms.clone();
        let client = self.client.clone();
        let since_token = Arc::clone(&self.since_token);
        let mut shutdown_rx = self.shutdown_rx.clone();
        let auto_accept = self.auto_accept_invites;

        // FIX #4: Do an initial sync to get the since token, skipping old messages.
        if since_token.read().await.is_none() {
            if let Some(token) = initial_sync(&client, &homeserver, access_token.as_str()).await {
                info!("Matrix: initial sync complete, skipping old messages");
                *since_token.write().await = Some(token);
            }
        }

        tokio::spawn(async move {
            let mut backoff = Duration::from_secs(1);

            loop {
                // Build /sync URL
                let since = since_token.read().await.clone();
                let mut url = format!(
                    "{}/_matrix/client/v3/sync?timeout={}&filter={{\"room\":{{\"timeline\":{{\"limit\":10}}}}}}",
                    homeserver, SYNC_TIMEOUT_MS
                );
                if let Some(ref token) = since {
                    url.push_str(&format!("&since={token}"));
                }

                let resp = tokio::select! {
                    _ = shutdown_rx.changed() => {
                        info!("Matrix adapter shutting down");
                        break;
                    }
                    result = client.get(&url).bearer_auth(access_token.as_str()).send() => {
                        match result {
                            Ok(r) => r,
                            Err(e) => {
                                warn!("Matrix sync error: {e}");
                                tokio::time::sleep(backoff).await;
                                backoff = (backoff * 2).min(Duration::from_secs(60));
                                continue;
                            }
                        }
                    }
                };

                if !resp.status().is_success() {
                    warn!("Matrix sync returned {}", resp.status());
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(60));
                    continue;
                }

                backoff = Duration::from_secs(1);

                let body: serde_json::Value = match resp.json().await {
                    Ok(b) => b,
                    Err(e) => {
                        warn!("Matrix sync parse error: {e}");
                        continue;
                    }
                };

                // Update since token
                if let Some(next) = body["next_batch"].as_str() {
                    *since_token.write().await = Some(next.to_string());
                }

                // FIX #1: Auto-accept room invites.
                if auto_accept {
                    if let Some(invites) = body["rooms"]["invite"].as_object() {
                        for (room_id, _invite_data) in invites {
                            if !allowed_rooms.is_empty()
                                && !allowed_rooms.iter().any(|r| r == room_id)
                            {
                                debug!(
                                    "Matrix: ignoring invite to {room_id} (not in allowed_rooms)"
                                );
                                continue;
                            }
                            accept_invite(&client, &homeserver, access_token.as_str(), room_id)
                                .await;
                        }
                    }
                }

                // Process room events
                if let Some(rooms) = body["rooms"]["join"].as_object() {
                    for (room_id, room_data) in rooms {
                        if !allowed_rooms.is_empty() && !allowed_rooms.iter().any(|r| r == room_id)
                        {
                            continue;
                        }

                        if let Some(events) = room_data["timeline"]["events"].as_array() {
                            for event in events {
                                let event_type = event["type"].as_str().unwrap_or("");
                                if event_type != "m.room.message" {
                                    continue;
                                }

                                let sender = event["sender"].as_str().unwrap_or("");
                                if sender == user_id {
                                    continue; // Skip own messages
                                }

                                // Check msgtype to handle different message types
                                let msgtype = event["content"]["msgtype"].as_str().unwrap_or("m.text");
                                let body = event["content"]["body"].as_str().unwrap_or("");

                                let msg_content = match msgtype {
                                    "m.image" => {
                                        // Matrix image message: extract mxc:// URL and convert to download URL
                                        let mxc_url = event["content"]["url"].as_str().unwrap_or("");
                                        if let Some(download_url) = mxc_to_download_url(&homeserver, mxc_url) {
                                            let caption = if body.is_empty() { None } else { Some(body.to_string()) };
                                            ChannelContent::Image { url: download_url, caption }
                                        } else {
                                            ChannelContent::Text(format!("[Image: {}]", body))
                                        }
                                    }
                                    "m.file" => {
                                        // Matrix file message
                                        let mxc_url = event["content"]["url"].as_str().unwrap_or("");
                                        let filename = event["content"]["filename"]
                                            .as_str()
                                            .or(Some(body))
                                            .unwrap_or("file")
                                            .to_string();
                                        if let Some(download_url) = mxc_to_download_url(&homeserver, mxc_url) {
                                            ChannelContent::File { url: download_url, filename }
                                        } else {
                                            ChannelContent::Text(format!("[File: {}]", filename))
                                        }
                                    }
                                    "m.video" => {
                                        // Matrix video message - treat as file
                                        let mxc_url = event["content"]["url"].as_str().unwrap_or("");
                                        let filename = if body.is_empty() { "video".to_string() } else { body.to_string() };
                                        if let Some(download_url) = mxc_to_download_url(&homeserver, mxc_url) {
                                            ChannelContent::File { url: download_url, filename }
                                        } else {
                                            ChannelContent::Text(format!("[Video: {}]", filename))
                                        }
                                    }
                                    "m.audio" => {
                                        // Matrix audio message - treat as file
                                        let mxc_url = event["content"]["url"].as_str().unwrap_or("");
                                        let filename = if body.is_empty() { "audio".to_string() } else { body.to_string() };
                                        if let Some(download_url) = mxc_to_download_url(&homeserver, mxc_url) {
                                            ChannelContent::File { url: download_url, filename }
                                        } else {
                                            ChannelContent::Text(format!("[Audio: {}]", filename))
                                        }
                                    }
                                    _ => {
                                        // Default: m.text or unknown - handle as text
                                        if body.is_empty() {
                                            continue;
                                        }
                                        if body.starts_with('/') {
                                            let parts: Vec<&str> = body.splitn(2, ' ').collect();
                                            let cmd = parts[0].trim_start_matches('/');
                                            let args: Vec<String> = parts
                                                .get(1)
                                                .map(|a| a.split_whitespace().map(String::from).collect())
                                                .unwrap_or_default();
                                            ChannelContent::Command {
                                                name: cmd.to_string(),
                                                args,
                                            }
                                        } else {
                                            ChannelContent::Text(body.to_string())
                                        }
                                    }
                                };

                                let event_id = event["event_id"].as_str().unwrap_or("").to_string();

                                // Use body for mention detection
                                let content = body;

                                // FIX #2: Detect @mentions in message text.
                                let mut metadata = HashMap::new();
                                if content.contains(&user_id) {
                                    metadata.insert(
                                        "was_mentioned".to_string(),
                                        serde_json::json!(true),
                                    );
                                }

                                // FIX #3: Determine if room is a DM (2 members) or group.
                                let is_group = get_room_member_count(
                                    &client,
                                    &homeserver,
                                    access_token.as_str(),
                                    room_id,
                                )
                                .await
                                .map(|count| count > 2)
                                .unwrap_or(true);

                                // For DMs, auto-set was_mentioned so dm_policy works.
                                if !is_group {
                                    metadata.insert(
                                        "was_mentioned".to_string(),
                                        serde_json::json!(true),
                                    );
                                    metadata.insert("is_dm".to_string(), serde_json::json!(true));
                                }

                                let channel_msg = ChannelMessage {
                                    channel: ChannelType::Matrix,
                                    platform_message_id: event_id,
                                    sender: ChannelUser {
                                        platform_id: room_id.clone(),
                                        display_name: sender.to_string(),
                                        openfang_user: None,
                                    },
                                    content: msg_content,
                                    target_agent: None,
                                    timestamp: Utc::now(),
                                    is_group,
                                    thread_id: None,
                                    metadata,
                                };

                                if tx.send(channel_msg).await.is_err() {
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    async fn send(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match content {
            ChannelContent::Text(text) => {
                // Detect if the text contains HTML tags (sent from bridge with MatrixHtml format)
                let has_html = contains_html_tags(&text);
                debug!(
                    "Matrix send: room={}, text_len={}, has_html={}",
                    user.platform_id,
                    text.len(),
                    has_html
                );

                if has_html {
                    debug!("Matrix send: detected HTML tags, sending as formatted message");
                    // Log first 300 chars of HTML for debugging (use char-level to avoid UTF-8 panic)
                    let preview = if text.chars().count() > 300 {
                        format!("{}...", text.chars().take(300).collect::<String>())
                    } else {
                        text.clone()
                    };
                    debug!("Matrix send: HTML content preview: {}", preview);

                    // Extract plain text fallback by stripping tags
                    let plain = strip_html_tags(&text);
                    debug!("Matrix send: plain text fallback len={}", plain.len());
                    self.api_send_html_message(&user.platform_id, &plain, &text)
                        .await?;
                } else {
                    debug!("Matrix send: no HTML tags detected, sending as plain text");
                    // Log why no HTML was detected - check for common patterns
                    let has_stars = text.contains("**");
                    let has_backticks = text.contains('`');
                    debug!(
                        "Matrix send: text analysis - has_stars={}, has_backticks={}, first 100 chars: {}",
                        has_stars,
                        has_backticks,
                        if text.chars().count() > 100 { text.chars().take(100).collect::<String>() } else { text.clone() }
                    );
                    self.api_send_message(&user.platform_id, &text).await?;
                }
            }
            _ => {
                self.api_send_message(&user.platform_id, "(Unsupported content type)")
                    .await?;
            }
        }
        Ok(())
    }

    async fn send_typing(&self, user: &ChannelUser) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/typing/{}",
            self.homeserver_url, user.platform_id, self.user_id
        );

        let body = serde_json::json!({
            "typing": true,
            "timeout": 5000,
        });

        let _ = self
            .client
            .put(&url)
            .bearer_auth(&*self.access_token)
            .json(&body)
            .send()
            .await;

        Ok(())
    }

    async fn send_reaction(
        &self,
        user: &ChannelUser,
        message_id: &str,
        reaction: &crate::types::LifecycleReaction,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Matrix uses m.reaction event type for emoji reactions
        // PUT /_matrix/client/v3/rooms/{roomId}/send/m.reaction/{txnId}
        let txn_id = uuid::Uuid::new_v4().to_string();
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/send/m.reaction/{}",
            self.homeserver_url, user.platform_id, txn_id
        );

        let body = serde_json::json!({
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": message_id,
                "key": reaction.emoji
            }
        });

        let resp = self
            .client
            .put(&url)
            .bearer_auth(&*self.access_token)
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            // Log but don't fail — reactions are non-critical
            tracing::debug!("Matrix reaction failed ({status}): {body}");
        }

        Ok(())
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.shutdown_tx.send(true);
        Ok(())
    }

    /// Send a streaming response using Matrix's edit API for real-time updates.
    ///
    /// Tool calls are shown FIRST (before content), using HTML formatting.
    /// Input/output limited to 3 lines max.
    async fn send_streaming(
        &self,
        user: &ChannelUser,
        mut rx: mpsc::Receiver<StreamEvent>,
        output_format: OutputFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut accumulated = String::new();
        let mut last_event_id: Option<String> = None;
        let mut last_edit_len = 0;
        let mut last_edit_time = std::time::Instant::now()
            .checked_sub(Duration::from_secs(10))
            .unwrap_or_else(std::time::Instant::now);

        // Track tool calls: (name, input_preview, output_preview, is_loading, is_error)
        let mut tool_calls: Vec<ToolCallInfo> = Vec::new();
        let mut current_tool_input: Option<String> = None;

        // Helper to build full message: tools FIRST, then content
        let build_message = |tools: &[ToolCallInfo], content: &str, fmt: OutputFormat| -> (String, String) {
            let content_html = formatter::format_for_channel(content, fmt);
            let tool_html = format_tool_calls_html(tools);
            let html = format!("{}{}", tool_html, content_html);
            let plain = format!("{}{}", strip_html_tags(&tool_html), strip_html_tags(&content_html));
            (html, plain)
        };

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::TextDelta { text } => {
                    accumulated.push_str(&text);

                    let now = std::time::Instant::now();
                    let chars_since_last = accumulated.chars().count().saturating_sub(last_edit_len);
                    let time_ok = now.duration_since(last_edit_time).as_millis() > EDIT_INTERVAL_MS as u128;
                    let content_ok = chars_since_last >= MIN_EDIT_CHARS;

                    if time_ok && content_ok {
                        let (html, plain) = build_message(&tool_calls, &accumulated, output_format);

                        if let Some(ref event_id) = last_event_id {
                            debug!("Matrix streaming: editing message, len={}", html.len());
                            let _ = self.api_edit_message(&user.platform_id, event_id, &plain, Some(&html)).await;
                        } else {
                            match self.api_send_html_message(&user.platform_id, &html, &plain).await {
                                Ok(event_id) => {
                                    if !event_id.is_empty() {
                                        last_event_id = Some(event_id);
                                    }
                                }
                                Err(e) => warn!("Matrix streaming: failed to send message: {e}"),
                            }
                        }

                        last_edit_len = accumulated.chars().count();
                        last_edit_time = now;
                    }
                }

                StreamEvent::ToolUseStart { name, .. } => {
                    tool_calls.push((name.clone(), None, None, true, false));
                    current_tool_input = Some(String::new());
                    debug!("Matrix streaming: tool start - {}", name);

                    let (html, plain) = build_message(&tool_calls, &accumulated, output_format);
                    if let Some(ref event_id) = last_event_id {
                        let _ = self.api_edit_message(&user.platform_id, event_id, &plain, Some(&html)).await;
                    } else {
                        match self.api_send_html_message(&user.platform_id, &html, &plain).await {
                            Ok(event_id) => {
                                if !event_id.is_empty() {
                                    last_event_id = Some(event_id);
                                }
                            }
                            Err(e) => warn!("Matrix streaming: failed to send: {e}"),
                        }
                    }
                }

                StreamEvent::ToolInputDelta { text } => {
                    if let Some(ref mut input) = current_tool_input {
                        input.push_str(&text);
                    }
                }

                StreamEvent::ToolUseEnd { name, input, .. } => {
                    for tool in &mut tool_calls {
                        if tool.0 == name && tool.3 {
                            let input_preview = if let Ok(json) = serde_json::to_string(&input) {
                                if json == "null" || json.is_empty() {
                                    let fallback = input.to_string();
                                    if fallback.is_empty() { NO_PARAMS.to_string() } else { fallback }
                                } else {
                                    json
                                }
                            } else {
                                let fallback = input.to_string();
                                if fallback.is_empty() { NO_PARAMS.to_string() } else { fallback }
                            };
                            tool.1 = Some(input_preview);
                            break;
                        }
                    }
                    current_tool_input = None;
                }

                StreamEvent::ToolExecutionResult { name, result_preview, is_error } => {
                    for tool in &mut tool_calls {
                        if tool.0 == name && tool.3 {
                            tool.2 = Some(result_preview.clone());
                            tool.3 = false;
                            tool.4 = is_error;
                            break;
                        }
                    }
                    debug!("Matrix streaming: tool result - {} (error={})", name, is_error);

                    let (html, plain) = build_message(&tool_calls, &accumulated, output_format);
                    if let Some(ref event_id) = last_event_id {
                        let _ = self.api_edit_message(&user.platform_id, event_id, &plain, Some(&html)).await;
                    }
                }

                StreamEvent::ContentComplete { .. } => {
                    let (html, plain) = build_message(&tool_calls, &accumulated, output_format);

                    if let Some(ref event_id) = last_event_id {
                        debug!("Matrix streaming: final edit, len={}", html.len());
                        self.api_edit_message(&user.platform_id, event_id, &plain, Some(&html)).await?;
                    } else {
                        debug!("Matrix streaming: sending as new message");
                        self.api_send_html_message(&user.platform_id, &html, &plain).await?;
                    }
                }

                StreamEvent::PhaseChange { phase, detail } => {
                    debug!("Matrix streaming: phase change to {} ({:?})", phase, detail);
                }

                _ => {}
            }
        }

        Ok(())
    }
}

/// Check if text contains HTML tags that should be sent as formatted Matrix message.
fn contains_html_tags(text: &str) -> bool {
    // Look for common HTML tags supported by Matrix
    let html_patterns = [
        "<b>", "</b>", "<i>", "</i>", "<u>", "</u>", "<strong>", "</strong>",
        "<em>", "</em>", "<del>", "</del>", "<code>", "</code>", "<pre>", "</pre>",
        "<a href", "</a>", "<blockquote>", "</blockquote>", "<h1>", "</h1>",
        "<h2>", "</h2>", "<h3>", "</h3>", "<h4>", "</h4>", "<h5>", "</h5>",
        "<h6>", "</h6>", "<ul>", "</ul>", "<ol>", "</ol>", "<li>", "</li>",
        "<table>", "</table>", "<thead>", "</thead>", "<tbody>", "</tbody>",
        "<tr>", "</tr>", "<th>", "</th>", "<td>", "</td>", "<p>", "</p>",
        "<br>", "<hr>", "<img", "<details>", "</details>", "<summary>", "</summary>",
    ];
    let lower = text.to_lowercase();
    html_patterns.iter().any(|tag| lower.contains(tag))
}

/// Strip HTML tags to produce plain text fallback.
fn strip_html_tags(html: &str) -> String {
    // Simple approach: remove tags and decode common entities
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let chars: Vec<char> = html.chars().collect();

    for i in 0..chars.len() {
        if chars[i] == '<' {
            in_tag = true;
        } else if chars[i] == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(chars[i]);
        }
    }

    // Decode common HTML entities
    result = result.replace("&amp;", "&");
    result = result.replace("&lt;", "<");
    result = result.replace("&gt;", ">");
    result = result.replace("&quot;", "\"");
    result = result.replace("&#39;", "'");
    result = result.replace("&nbsp;", " ");

    // Collapse multiple whitespace
    let mut collapsed = String::new();
    let mut prev_whitespace = false;
    for c in result.chars() {
        if c.is_whitespace() {
            if !prev_whitespace {
                collapsed.push(c);
            }
            prev_whitespace = true;
        } else {
            collapsed.push(c);
            prev_whitespace = false;
        }
    }

    collapsed.trim().to_string()
}

/// Get icon for a tool based on its name.
fn tool_icon(name: &str) -> &'static str {
    match name {
        "web_search" | "web_fetch" => "🌐",
        "read_file" | "glob" | "grep" | "search_file_content" => "📄",
        "write_file" | "replace" => "✏️",
        "run_shell_command" | "bash" => "💻",
        "image_generate" | "image_read" => "🖼️",
        "web_search_planning" => "🔍",
        "list_directory" => "📁",
        "todo_write" | "todo_read" => "📋",
        "ask_user_question" => "❓",
        _ => "🔧",
    }
}

/// Truncate text to max lines and max characters.
fn truncate_text(text: &str, max_lines: usize, max_chars: usize) -> String {
    let mut result = String::new();
    let mut char_count = 0;

    for line in text.lines().take(max_lines) {
        if char_count >= max_chars {
            break;
        }
        if !result.is_empty() {
            result.push('\n');
            char_count += 1;
        }
        let remaining = max_chars.saturating_sub(char_count);
        if line.chars().count() > remaining {
            let truncated: String = line.chars().take(remaining).collect();
            result.push_str(&truncated);
            char_count = max_chars;
            break;
        } else {
            result.push_str(line);
            char_count += line.chars().count();
        }
    }

    let needs_ellipsis = text.lines().count() > max_lines
        || text.chars().count() > max_chars
        || char_count >= max_chars;

    if needs_ellipsis {
        format!("{}...", result)
    } else {
        result
    }
}

/// Escape HTML special characters.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Tool call info for formatting.
type ToolCallInfo = (String, Option<String>, Option<String>, bool, bool);

/// Format completed tool calls as HTML.
fn format_tool_calls_html(tools: &[ToolCallInfo]) -> String {
    let completed_tools: Vec<_> = tools.iter()
        .filter(|(_, _, output, _, _)| output.is_some())
        .collect();

    if completed_tools.is_empty() {
        return String::new();
    }

    let mut html = String::new();
    let count = completed_tools.len();

    // Header
    html.push_str(&format!("<b>{} ({})</b><br/>", TOOL_CALLS_HEADER, count));

    for (idx, (name, input, output, _is_loading, is_error)) in completed_tools.iter().enumerate() {
        let icon = tool_icon(name);

        // Wrap entire tool in blockquote for indentation
        html.push_str("<blockquote>");
        html.push_str(&format!("{}. {} <b>{}</b>", idx + 1, icon, name));

        // Input section
        if let Some(ref inp) = input {
            let truncated = truncate_text(inp, MAX_PREVIEW_LINES, MAX_INPUT_CHARS);
            let indented = truncated.lines()
                .map(|line| format!("{}{}", INDENT_PREFIX, escape_html(line)))
                .collect::<Vec<_>>()
                .join("<br/>");
            html.push_str(&format!(
                "<br/><br/><blockquote>{}<br/><code>{}</code></blockquote>",
                INPUT_LABEL, indented
            ));
        }

        // Output section
        if let Some(ref out) = output {
            let truncated = truncate_text(out, MAX_PREVIEW_LINES, MAX_OUTPUT_CHARS);
            let status_icon = if *is_error { ERROR_LABEL } else { "" };
            let label = if *is_error { ERROR_LABEL } else { OUTPUT_LABEL };
            let indented = truncated.lines()
                .map(|line| format!("{}{}", INDENT_PREFIX, escape_html(line)))
                .collect::<Vec<_>>()
                .join("<br/>");
            html.push_str(&format!(
                "<blockquote>{} {}<br/><code>{}</code></blockquote>",
                status_icon, label, indented
            ));
        }

        html.push_str("</blockquote>");
    }

    html.push_str("<br/>");
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_adapter_creation() {
        let adapter = MatrixAdapter::new(
            "https://matrix.org".to_string(),
            "@bot:matrix.org".to_string(),
            "access_token".to_string(),
            vec![],
        );
        assert_eq!(adapter.name(), "matrix");
    }

    #[test]
    fn test_matrix_allowed_rooms() {
        let adapter = MatrixAdapter::new(
            "https://matrix.org".to_string(),
            "@bot:matrix.org".to_string(),
            "token".to_string(),
            vec!["!room1:matrix.org".to_string()],
        );
        assert!(adapter.is_allowed_room("!room1:matrix.org"));
        assert!(!adapter.is_allowed_room("!room2:matrix.org"));

        let open = MatrixAdapter::new(
            "https://matrix.org".to_string(),
            "@bot:matrix.org".to_string(),
            "token".to_string(),
            vec![],
        );
        assert!(open.is_allowed_room("!any:matrix.org"));
    }

    #[test]
    fn test_contains_html_tags() {
        assert!(contains_html_tags("<b>bold</b>"));
        assert!(contains_html_tags("<p>paragraph</p>"));
        assert!(contains_html_tags("<a href=\"url\">link</a>"));
        assert!(contains_html_tags("<pre><code>code</code></pre>"));
        assert!(contains_html_tags("<ul><li>item</li></ul>"));
        assert!(!contains_html_tags("plain text"));
        assert!(!contains_html_tags("2 < 3 and 4 > 1")); // Not HTML tags
    }

    #[test]
    fn test_strip_html_tags() {
        assert_eq!(strip_html_tags("<b>bold</b>"), "bold");
        assert_eq!(strip_html_tags("<p>Hello <strong>world</strong>!</p>"), "Hello world!");
        assert_eq!(strip_html_tags("<a href=\"https://example.com\">link</a>"), "link");
        // Note: strip_html_tags simply removes tags without adding spacing
        assert_eq!(
            strip_html_tags("<ul><li>one</li><li>two</li></ul>"),
            "onetwo"
        );
        assert_eq!(strip_html_tags("&amp; &lt; &gt;"), "& < >");
        assert_eq!(strip_html_tags("plain text"), "plain text");
    }
}
