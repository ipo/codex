use codex_protocol::items::TurnItem;
use std::collections::HashMap;

pub(super) struct PresentationItem {
    pub turn_item: TurnItem,
    pub streamed_to_client: bool,
}

#[derive(Default)]
pub(super) struct PresentationLifecycle {
    items: HashMap<String, PresentationItem>,
    legacy_active_item_id: Option<String>,
}

impl PresentationLifecycle {
    pub fn legacy_active_item(&self) -> Option<&TurnItem> {
        self.legacy_active_item_id
            .as_deref()
            .and_then(|item_id| self.items.get(item_id))
            .map(|item| &item.turn_item)
    }

    pub fn insert(&mut self, item: PresentationItem) -> bool {
        let item_id = item.turn_item.id();
        self.legacy_active_item_id = Some(item_id.clone());
        if self.items.contains_key(&item_id) {
            return false;
        }
        self.items.insert(item_id, item);
        true
    }

    pub fn get(&self, item_id: Option<&str>) -> Option<&PresentationItem> {
        let item_id = item_id.or(self.legacy_active_item_id.as_deref())?;
        self.items.get(item_id)
    }

    pub fn take(&mut self, item_id: Option<&str>) -> Option<PresentationItem> {
        let item_id = item_id
            .map(str::to_string)
            .or_else(|| self.legacy_active_item_id.clone())?;
        if self.legacy_active_item_id.as_deref() == Some(item_id.as_str()) {
            self.legacy_active_item_id = None;
        }
        self.items.remove(&item_id)
    }
}
