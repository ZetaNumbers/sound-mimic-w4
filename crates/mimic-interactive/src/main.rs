use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_systems(Update, update_ui)
        .run();
}

fn update_ui(mut contexts: EguiContexts) {
    egui::Window::new("Title").show(contexts.ctx_mut(), |ui| {
        ui.label("hello world");
    });
}
