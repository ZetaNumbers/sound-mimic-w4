use anyhow::Context;

pub struct Cartridge<E> {
    update: wt::TypedFunc<(), ()>,
    store: wt::Store<State<E>>,
}

impl<E: Engine> Cartridge<E> {
    pub fn new(bytes: impl AsRef<[u8]>, engine: E) -> anyhow::Result<Self> {
        fn impl_<E: Engine>(bytes: &[u8], state: E) -> anyhow::Result<Cartridge<E>> {
            let state = State { engine: state };
            let engine = wt::Engine::default();
            let mut store = wt::Store::new(&engine, state);
            let module = wt::Module::new(&engine, bytes)?;

            let mut linker = wt::Linker::new(&engine);
            linker.define_unknown_imports_as_default_values(&module)?;
            linker.allow_shadowing(true);
            linker.func_wrap(
                "env",
                "tone",
                |mut caller: wt::Caller<'_, State<E>>,
                 frequency: u32,
                 duration: u32,
                 volume: u32,
                 flags: u32|
                 -> anyhow::Result<()> {
                    caller
                        .data_mut()
                        .engine
                        .tone(frequency, duration, volume, flags)
                },
            )?;

            let mem = wt::Memory::new(&mut store, wt::MemoryType::new(1, Some(1)))?;
            linker.define(&mut store, "env", "memory", mem)?;

            let instance = linker.instantiate(&mut store, &module)?;
            let start = instance
                .get_func(&mut store, "start")
                .unwrap_or_else(|| wt::Func::wrap(&mut store, || ()))
                .typed::<(), ()>(&mut store)
                .context("`start` funtion has wrong type, `fn() -> ()` expected")?;
            let update = instance
                .get_func(&mut store, "update")
                .unwrap_or_else(|| wt::Func::wrap(&mut store, || ()))
                .typed::<(), ()>(&mut store)
                .context("`update` funtion has wrong type, `fn() -> ()` expected")?;

            store.data_mut().engine.before_start()?;
            start.call(&mut store, ())?;
            store.data_mut().engine.after_start()?;

            Ok(Cartridge { update, store })
        }

        impl_(bytes.as_ref(), engine)
    }

    pub fn update(&mut self) -> anyhow::Result<()> {
        self.store.data_mut().engine.before_update()?;
        self.update.call(&mut self.store, ())?;
        self.store.data_mut().engine.after_update()
    }
}

impl<E> Cartridge<E> {
    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.store.data_mut().engine
    }
}

pub trait Engine {
    fn tone(
        &mut self,
        _frequency: u32,
        _duration: u32,
        _volume: u32,
        _flags: u32,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    fn before_start(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    fn after_start(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    fn before_update(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
    fn after_update(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[derive(Clone)]
struct State<E> {
    engine: E,
}
