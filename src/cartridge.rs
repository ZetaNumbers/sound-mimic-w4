use crate::audio::Audio;

pub struct Cartridge {
    update: wt::TypedFunc<(), ()>,
    store: wt::Store<State>,
}

impl Cartridge {
    pub fn new(bytes: impl AsRef<[u8]>) -> Self {
        fn impl_(bytes: &[u8]) -> Cartridge {
            let engine = wt::Engine::default();

            let mut store = wt::Store::new(&engine, State::default());
            let mem = wt::Memory::new(&mut store, wt::MemoryType::new(1, Some(1))).unwrap();

            let module = wt::Module::new(&engine, bytes).unwrap();

            let mut linker = wt::Linker::new(&engine);
            linker
                .define_unknown_imports_as_default_values(&module)
                .unwrap();
            linker.allow_shadowing(true);
            linker
                .func_wrap(
                    "env",
                    "tone",
                    |mut caller: wt::Caller<'_, State>,
                     frequency: u32,
                     duration: u32,
                     volume: u32,
                     flags: u32| {
                        caller
                            .data_mut()
                            .audio
                            .tone(frequency, duration, volume, flags);
                    },
                )
                .unwrap();

            linker.define(&mut store, "env", "memory", mem).unwrap();

            let instance = linker.instantiate(&mut store, &module).unwrap();
            let start = instance
                .get_typed_func::<(), ()>(&mut store, "start")
                .unwrap();
            let update = instance
                .get_typed_func::<(), ()>(&mut store, "update")
                .unwrap();

            start.call(&mut store, ()).unwrap();

            Cartridge { update, store }
        }

        impl_(bytes.as_ref())
    }

    pub fn update(&mut self) {
        self.update.call(&mut self.store, ()).unwrap();
    }

    pub fn audio_mut(&mut self) -> &mut Audio {
        &mut self.store.data_mut().audio
    }
}

#[derive(Default)]
struct State {
    audio: Audio,
}
