use bevy::prelude::*;
use bevy_egui::{
    EguiContexts, EguiPlugin,
    egui::{self, Widget},
};
use bevy_vello::{VelloPlugin, prelude::*};
use rustfft::{FftDirection, num_complex::Complex};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .add_plugins(VelloPlugin::default())
        .add_systems(Startup, (setup_vector_graphics, setup_ui))
        .add_systems(Update, draw_graph)
        .run();
}

#[derive(Component)]
struct DrawOptions {
    time: usize,
    play: bool,
    zoom: bool,
    dist: bool,
    dist_min_of_n: i32,
    dist_mode: DistMode,
    dist_window_size: usize,
    fft: bool,
    fft_limit_const: bool,
    fft_avg_window: usize,
}

impl DrawOptions {
    fn new(len: usize) -> Self {
        Self {
            time: len,
            play: false,
            zoom: true,
            dist: false,
            dist_min_of_n: 1,
            dist_mode: DistMode::default(),
            dist_window_size: 3,
            fft: false,
            fft_limit_const: false,
            fft_avg_window: 1,
        }
    }
}

#[derive(PartialEq, Eq, Default, Clone, Copy)]
enum DistMode {
    #[default]
    Cumulative,
    Probability,
}

impl DistMode {
    fn as_str(self) -> &'static str {
        match self {
            DistMode::Cumulative => "Cumulative",
            DistMode::Probability => "Probability",
        }
    }
}

fn setup_vector_graphics(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
    commands.spawn(VelloSceneBundle::default());
}

fn setup_ui(mut commands: Commands) {
    commands.spawn(DrawOptions::new(ERRORS.len()));
}

fn draw_graph(
    mut contexts: EguiContexts,
    mut query_scene: Query<&mut VelloScene>,
    mut query_ui: Query<&mut DrawOptions>,
) {
    let mut scene = query_scene.single_mut();
    let mut draw_opts = query_ui.single_mut();
    // Reset scene every frame
    *scene = VelloScene::default();

    egui::Window::new("Draw Options").show(contexts.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            let label = ui.label("Time");
            let min_time = draw_opts.fft as usize * draw_opts.fft_avg_window
                + draw_opts.dist as usize * draw_opts.dist_window_size
                + 4;
            egui::Slider::new(&mut draw_opts.time, min_time..=ERRORS.len())
                .ui(ui)
                .labelled_by(label.id);
            let at_the_end = draw_opts.time >= ERRORS.len();
            draw_opts.play ^= ui
                .small_button(if draw_opts.play {
                    "⏸"
                } else if at_the_end {
                    "↺"
                } else {
                    "⏵"
                })
                .clicked();
            if draw_opts.play & at_the_end {
                draw_opts.time = min_time;
            }
        });
        let mut left_samples = draw_opts.time;
        ui.checkbox(&mut draw_opts.zoom, "Normalize samples");
        ui.separator();
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.checkbox(&mut draw_opts.dist, "Distribution analysis");

                if !draw_opts.dist {
                    ui.disable();
                }
                ui.horizontal(|ui| {
                    let label = ui.label("Samples for minimum");
                    egui::DragValue::new(&mut draw_opts.dist_min_of_n)
                        .range(1..=i32::MAX)
                        .speed(0.1)
                        .ui(ui)
                        .labelled_by(label.id);
                });
                // egui::ComboBox::from_label("mode")
                //     .selected_text(draw_opts.dist_mode.as_str())
                //     .show_ui(ui, |ui| {
                //         ui.selectable_value(
                //             &mut draw_opts.dist_mode,
                //             DistMode::Cumulative,
                //             DistMode::Cumulative.as_str(),
                //         );
                //         ui.selectable_value(
                //             &mut draw_opts.dist_mode,
                //             DistMode::Probability,
                //             DistMode::Probability.as_str(),
                //         );
                //     });

                match draw_opts.dist_mode {
                    DistMode::Cumulative => {
                        ui.disable();
                        if draw_opts.dist {
                            draw_opts.fft = false
                        }
                    }
                    DistMode::Probability => (),
                }
                // ui.horizontal(|ui| {
                //     let label = ui.label("Regression window");
                //     egui::DragValue::new(&mut draw_opts.dist_window_size)
                //         .ui(ui)
                //         .labelled_by(label.id);
                //     draw_opts.dist_window_size = draw_opts.dist_window_size.clamp(3, left_samples);
                //     left_samples -= draw_opts.dist_window_size;
                // });
            });
            ui.separator();
            ui.vertical(|ui| {
                if draw_opts.dist && draw_opts.dist_mode == DistMode::Cumulative {
                    ui.disable();
                }
                ui.checkbox(&mut draw_opts.fft, "FFT");
                if !draw_opts.fft {
                    ui.disable();
                }
                ui.checkbox(&mut draw_opts.fft_limit_const, "Limit constant");
                ui.horizontal(|ui| {
                    ui.label("Averaging window");
                    egui::DragValue::new(&mut draw_opts.fft_avg_window).ui(ui);
                    draw_opts.fft_avg_window = draw_opts.fft_avg_window.clamp(1, left_samples);
                    left_samples -= draw_opts.fft_avg_window;
                });
            });
        });
    });

    let mut errors = ERRORS[..draw_opts.time].to_vec();
    if draw_opts.play {
        draw_opts.time += 1;
    }

    if draw_opts.dist {
        errors.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("failed sorting for distribution analysis")
        });

        if draw_opts.dist_mode == DistMode::Probability {
            let n = draw_opts.dist_window_size;
            errors = errors
                .windows(n)
                .map(|w| {
                    let sxy: f64 = w.iter().enumerate().map(|(x, &y)| (x + 1) as f64 * y).sum();
                    let sx = (n * (n + 1) / 2) as f64;
                    let sy: f64 = w.iter().copied().sum();
                    let sx2 = (1..n + 1).map(|x| x * x).sum::<usize>() as f64;

                    (n as f64 * sx2 - sx * sx) / (n as f64 * sxy - sx * sy)
                })
                .collect()
        }
    }

    if draw_opts.fft {
        let mut cerrors: Vec<_> = errors.iter().map(|r| Complex::from(*r)).collect();
        let mut fft_planner = rustfft::FftPlanner::new();
        let fft = fft_planner.plan_fft(cerrors.len(), FftDirection::Forward);
        fft.process(&mut cerrors);
        let errors_temp: Vec<_> = cerrors.iter().map(|&c| c.norm()).collect();
        errors = errors_temp
            .windows(draw_opts.fft_avg_window)
            .map(|w| w.iter().copied().sum::<f64>() / w.len() as f64)
            .collect();
    }

    let min_y = if draw_opts.zoom {
        *errors.iter().min_by(cmp_or_panic).unwrap()
    } else {
        0.0
    };
    let max_y = *errors[(draw_opts.fft && !draw_opts.fft_limit_const) as usize..]
        .iter()
        .max_by(cmp_or_panic)
        .unwrap();
    let min_x = 0.0;
    let max_x = errors.len() as f64;

    if draw_opts.fft {
        let midpoint = errors.len().div_ceil(2);
        let (pos, neg) = errors.split_at(midpoint);
        let mut new_errors = Vec::with_capacity(errors.len());
        new_errors.extend_from_slice(neg);
        new_errors.extend_from_slice(pos);
        errors = new_errors;
    }

    let mut path: Vec<_> = errors
        .iter()
        .enumerate()
        .map(|(x, &y)| {
            kurbo::PathEl::LineTo(kurbo::Point {
                x: (2.0 * (x as f64 - min_x) / (max_x - min_x) - 1.0).clamp(-1.0, 1.0),
                y: (2.0 * (y - min_y) / (max_y - min_y) - 1.0).clamp(-1.0, 1.0),
            })
        })
        .collect();
    if draw_opts.dist && draw_opts.dist_mode == DistMode::Cumulative && !draw_opts.fft {
        path.iter_mut().for_each(|p| {
            let kurbo::PathEl::LineTo(p) = p else {
                unreachable!()
            };
            *p = kurbo::Point {
                x: p.y,
                y: 1.0 - (0.5 - p.x * 0.5).powi(draw_opts.dist_min_of_n) * 2.0,
            };
        });
    }

    // Animate the corner radius
    scene.stroke(
        &kurbo::Stroke::new(0.0015),
        kurbo::Affine::scale_non_uniform(625.0, -350.0),
        peniko::Color::rgb(-1.0, 1.0, 1.0),
        None,
        &path.as_slice(),
    );

    // transform.rotation = Quat::from_rotation_z(-std::f32::consts::TAU * sin_time);
}

fn cmp_or_panic(y1: &&f64, y2: &&f64) -> std::cmp::Ordering {
    f64::partial_cmp(y1, y2).unwrap()
}

const ERRORS: &[f64] = &[
    1.0629038, 1.0629208, 1.063053, 1.0630442, 1.0624775, 1.0619497, 1.0616138, 1.0613134,
    1.0611763, 1.0609347, 1.0609349, 1.0611612, 1.0614297, 1.0616721, 1.0618542, 1.0621214,
    1.0621761, 1.0623032, 1.0623659, 1.0624789, 1.0626297, 1.0625266, 1.062477, 1.0625464,
    1.0626495, 1.0627042, 1.0626259, 1.0624703, 1.062351, 1.061994, 1.0614445, 1.0608943,
    1.0604634, 1.0602552, 1.0604002, 1.0604678, 1.0604734, 1.0603302, 1.060129, 1.0599539,
    1.0597478, 1.0598298, 1.0603172, 1.061072, 1.061797, 1.062334, 1.0626053, 1.0626186, 1.0624113,
    1.0621251, 1.0617783, 1.0615379, 1.0614148, 1.0614982, 1.0617429, 1.0620298, 1.0620121,
    1.0618818, 1.0615728, 1.0612379, 1.0608952, 1.0606432, 1.0605719, 1.0607318, 1.0609621,
    1.0610201, 1.0613128, 1.0616696, 1.0621473, 1.0626532, 1.0631659, 1.063492, 1.0636888,
    1.0638508, 1.0641712, 1.0644392, 1.0647572, 1.0649376, 1.0648998, 1.0647533, 1.0646254,
    1.0643312, 1.0637392, 1.0631115, 1.0625274, 1.062305, 1.0623313, 1.0623902, 1.0626904,
    1.0630974, 1.0632092, 1.0630077, 1.062829, 1.0626749, 1.0626872, 1.0631437, 1.0638926,
    1.064828, 1.0654714, 1.0658041, 1.0657715, 1.0653635, 1.0648041, 1.0643353, 1.0640596,
    1.0639694, 1.0641387, 1.064262, 1.0645096, 1.0645839, 1.0644084, 1.064294, 1.0641237,
    1.0638156, 1.0635738, 1.0636203, 1.063545, 1.0635645, 1.0638155, 1.064221, 1.0646777, 1.065061,
    1.0656168, 1.0662225, 1.0665158, 1.066653, 1.0666797, 1.0667496, 1.0668731, 1.0670222,
    1.0672144, 1.067446, 1.0674939, 1.0672463, 1.0666602, 1.0656756, 1.0647329, 1.0639194,
    1.0632544, 1.06284, 1.0627962, 1.062994, 1.0631111, 1.0628641, 1.062787, 1.0626742, 1.0626862,
    1.0626353, 1.0628855, 1.0633339, 1.0637203, 1.0639863, 1.0641122, 1.0641071, 1.0638874,
    1.0633539, 1.0627581, 1.0622602, 1.0619295, 1.0616449, 1.0613256, 1.061096, 1.0605887,
    1.0601858, 1.059797, 1.059708, 1.0597752, 1.0597674, 1.0597208, 1.0599533, 1.0602303,
    1.0604728, 1.0605319, 1.0606198, 1.0607777, 1.0610359, 1.0612514, 1.0613294, 1.0614944,
    1.061858, 1.0622466, 1.0624021, 1.0623835, 1.0626569, 1.0628835, 1.0630741, 1.0632216,
    1.0629768, 1.0625947, 1.0617832, 1.0607651, 1.0599762, 1.0595323, 1.0594895, 1.0594757,
    1.0598997, 1.060491, 1.0610234, 1.0615907, 1.0618751, 1.0622011, 1.0624789, 1.0627921,
    1.0630728, 1.0631456, 1.0630094, 1.0629157, 1.0626143, 1.0620995, 1.061543, 1.0610299,
    1.0604099, 1.0599035, 1.0592833, 1.0588068, 1.0582004, 1.0579618, 1.058029, 1.0580618,
    1.0580328, 1.0581565, 1.0585139, 1.0589736, 1.0590864, 1.0589956, 1.0588183, 1.0587966,
    1.0591106, 1.0596516, 1.0602002, 1.060754, 1.0611674, 1.0614417, 1.061842, 1.0622483, 1.062688,
    1.0631722, 1.0635504, 1.0639696, 1.0642061, 1.0642416, 1.0636363, 1.0627594, 1.0619466,
    1.0613745, 1.060863, 1.0603993, 1.0603378, 1.060399, 1.060489, 1.0607662, 1.0613798, 1.0620996,
    1.0628846, 1.0636812, 1.0642291, 1.0644474, 1.0643166, 1.0640517, 1.0638216, 1.0635837,
    1.0635144, 1.0635182, 1.0633588, 1.0630816, 1.0627868, 1.0624669, 1.061918, 1.0616455,
    1.0615916, 1.0613478, 1.0612034, 1.0610343, 1.061138, 1.0613176, 1.0614107, 1.0614294,
    1.0614592, 1.061426, 1.0613941, 1.061584, 1.0617528, 1.0619687, 1.0620172, 1.0621396,
    1.0625442, 1.0629088, 1.0630405, 1.063084, 1.0632871, 1.0636177, 1.063881, 1.0640104,
    1.0639739, 1.0638372, 1.0635132, 1.0630745, 1.0624814, 1.0619707, 1.0614315, 1.0608134,
    1.0603421, 1.0602567, 1.0605847, 1.06116, 1.0619905, 1.0626024, 1.0630962, 1.0634619,
    1.0635892, 1.063221, 1.0627502, 1.0622106, 1.0616503, 1.0612278, 1.061063, 1.0610951,
    1.0606271, 1.0599484, 1.059244, 1.0586905, 1.0581934, 1.0578368, 1.0577258, 1.0578325,
    1.058302, 1.0588254, 1.0591804, 1.0591135, 1.0591573, 1.0593057, 1.0595113, 1.0595543,
    1.059424, 1.0595187, 1.0597245, 1.0599324, 1.060406, 1.0608304, 1.0610781, 1.0612806,
    1.0615997, 1.0620149, 1.0623813, 1.0626372, 1.0627872, 1.0629331, 1.0631806, 1.063297,
    1.0634376, 1.0636307, 1.0635676, 1.0632814, 1.0629106, 1.0623147, 1.0620276, 1.0622516,
    1.0627487, 1.0634731, 1.0642817, 1.0652392, 1.0661005, 1.066539, 1.0664445, 1.0659807,
    1.0653784, 1.0649028, 1.0646182, 1.0640644, 1.0633012, 1.0627111, 1.0622398, 1.0616075,
    1.0609229, 1.0603435, 1.059978, 1.0597459, 1.0596979, 1.0597888, 1.0596467, 1.0595227,
    1.0593655, 1.0592214, 1.0591196, 1.0591196, 1.0594093, 1.0594352, 1.0590898, 1.058904,
    1.0590322, 1.0593426, 1.0597938, 1.060279, 1.0607904, 1.061019, 1.0614214, 1.0615842,
    1.0618623, 1.0621862, 1.0625346, 1.0627202, 1.062802, 1.062737, 1.0626372, 1.062443, 1.0623158,
    1.0621003, 1.0619203, 1.0619255, 1.062145, 1.062737, 1.0636714, 1.0648581, 1.0658903,
    1.0669115, 1.067611, 1.0679153, 1.0679733, 1.067812, 1.0673484, 1.0665855, 1.0658756,
    1.0650494, 1.0642046, 1.0634497, 1.0629299, 1.0626465, 1.0625018, 1.0623332, 1.0623734,
    1.0626335, 1.0627643, 1.0627143, 1.0626607, 1.0626669, 1.0626308, 1.0623869, 1.0622958,
    1.0625101, 1.0627556, 1.0628644, 1.0628554, 1.0628282, 1.062885, 1.0630943, 1.0632725,
    1.0636154, 1.0639652, 1.0642598, 1.0643378, 1.0644755, 1.0646054, 1.0647919, 1.0650444,
    1.0652268, 1.0651064, 1.06466, 1.064217, 1.0640241, 1.0639045, 1.0637448, 1.0637816, 1.0639483,
    1.0643588, 1.0648187, 1.0652543, 1.065523, 1.0657581, 1.0657665, 1.0658175, 1.0657748,
    1.0652174, 1.0643862, 1.0633872, 1.062396, 1.0615089, 1.060772, 1.0604033, 1.0603163,
    1.0603653, 1.0605352, 1.0609874, 1.061688, 1.0624292, 1.0632156, 1.0637182, 1.0640134,
    1.064008, 1.0636933, 1.0634459, 1.0634093, 1.0634646, 1.0636519, 1.0637985, 1.0640856,
    1.0645628, 1.0648167, 1.0647306, 1.0647199, 1.0646203, 1.0645406, 1.0644814, 1.064658,
    1.0649846, 1.06513, 1.0655359, 1.0659941, 1.0661792, 1.0658959, 1.0654899, 1.0652635,
    1.0650705, 1.0649139, 1.0649434, 1.0649765, 1.0653665, 1.0658305, 1.0661255, 1.0660547,
    1.0656897, 1.065403, 1.065316, 1.0650666, 1.064644, 1.0641134, 1.0635781, 1.062907, 1.0621878,
    1.0614027, 1.0608004, 1.0602505, 1.0600164, 1.0598983, 1.0599679, 1.0603338, 1.060686,
    1.0611261, 1.0614649, 1.0617529, 1.0618635, 1.0618278, 1.0616021, 1.0614202, 1.0611405,
    1.0609647, 1.0609282, 1.0608281, 1.061041, 1.0611583, 1.0611295, 1.061166, 1.0611305,
    1.0609212, 1.0607104, 1.0605608, 1.0605569, 1.0608263, 1.0614519, 1.0621353, 1.0624037,
    1.0625129, 1.0625669, 1.0626084, 1.0625782, 1.0625653, 1.0628154, 1.0631828, 1.0638655,
    1.064769, 1.0656841, 1.0663985, 1.0666425, 1.0666689, 1.0666761, 1.066514, 1.0663131,
    1.0662355, 1.0659721, 1.0656235, 1.0655026, 1.0653522, 1.0650092, 1.0645349, 1.0639455,
    1.0632554, 1.0626261, 1.0622464, 1.0620153, 1.0619951, 1.0622188, 1.0624859, 1.0628154,
    1.0629476, 1.0628953, 1.0625029, 1.0619756, 1.0612822, 1.0607306, 1.0603819, 1.0601579,
    1.0600785, 1.0602701, 1.0605786, 1.0608019, 1.0607791, 1.0605377, 1.0601932, 1.0600319,
    1.0601833, 1.0605766, 1.06104, 1.0613252, 1.0614911, 1.0615487, 1.061644, 1.0619043, 1.0623281,
    1.062573, 1.0627463, 1.0630344, 1.063584, 1.0641172, 1.0641483, 1.0642822, 1.0644983,
    1.0645497, 1.0644683, 1.0641371, 1.0639051, 1.0636415, 1.0633204, 1.0633184, 1.0635998,
    1.063823, 1.0637797, 1.0637882, 1.0633513, 1.0627049, 1.0622673, 1.0620372, 1.0618387,
    1.0615947, 1.0615624, 1.0616828, 1.0619954, 1.0622287, 1.062457, 1.0623996, 1.0620588,
    1.0614594, 1.0608379, 1.0601938, 1.0596721, 1.0593508, 1.0592849, 1.0595729, 1.0601039,
    1.0604397, 1.0606171, 1.0606786, 1.0605159, 1.0603969, 1.0603642, 1.0600463, 1.0598347,
    1.059837, 1.0600348, 1.0601354, 1.0602707, 1.0607321, 1.0614054, 1.0620313, 1.0624295,
    1.0622556, 1.0618001, 1.0614198, 1.0612314, 1.0608306, 1.0608114, 1.0606774, 1.0601817,
    1.0595709, 1.0589844, 1.0587238, 1.0589147, 1.0594407, 1.0597008, 1.0602173, 1.0607128,
    1.0608908, 1.0609015, 1.0606667, 1.0604037, 1.0600457, 1.0597866, 1.0596977, 1.059674,
    1.0597106, 1.0598563, 1.0597508, 1.0591853, 1.0584915, 1.0577971, 1.0572339, 1.0568645,
    1.0566758, 1.0568564, 1.0574871, 1.0582591, 1.0588188, 1.0592569, 1.0597563, 1.0603964,
    1.0610381, 1.0614322, 1.0612642, 1.0608095, 1.0605402, 1.0605274, 1.0605006, 1.0606747,
    1.0609301, 1.0613056, 1.0616484, 1.0619737, 1.0621148, 1.0619808, 1.0617318, 1.0614126,
    1.0612838, 1.0614772, 1.061497, 1.0614204, 1.0612706, 1.0610592, 1.0606129, 1.060512,
    1.0718255, 1.0719662, 1.0722215, 1.0726653, 1.0730776, 1.0733074, 1.0733912, 1.0732495,
    1.0727297, 1.0722927, 1.0721492, 1.0720972, 1.0721427, 1.0721704, 1.0720332, 1.0716925,
    1.07132, 1.0710526, 1.0709504, 1.0707328, 1.0704316, 1.0703202, 1.0704288, 1.0707746, 1.071305,
    1.071971, 1.0724692, 1.0727563, 1.0729672, 1.0731817, 1.0733597, 1.0732614, 1.073298,
    1.0734626, 1.0736666, 1.0738755, 1.0740272, 1.074168, 1.0742915, 1.074518, 1.0747199,
    1.0747511, 1.0745898, 1.0742797, 1.0740826, 1.0739408, 1.0738697, 1.0738763, 1.0738349,
    1.0735714, 1.0735136, 1.0736774, 1.0737532, 1.0739682, 1.0745069, 1.074914, 1.0753441,
    1.075414, 1.0753273, 1.0751023, 1.0747077, 1.0743834, 1.0742834,
];
