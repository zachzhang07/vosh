import argparse
from nerf.utils import *
import time

if __name__ == '__main__':
    start_t = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    parser.add_argument('--vol_path', type=str)
    parser.add_argument('--vert_offset', action='store_true')
    parser.add_argument('--lr_vert', type=float, default=1e-4,
                        help="initial learning ratio for vert optimization")
    parser.add_argument('--no_normal', action='store_true')
    parser.add_argument('--ec_center', type=float, default=0.5, help="loss scale")
    parser.add_argument('--vis_error', action='store_true', help="visualize error grid based on mesh")

    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--fast_baking', action='store_true',
                        help="faster baking at the cost of maybe missing blocks at background")

    ### new options
    parser.add_argument('--render', type=str, default='mesh', choices=['mesh'])
    parser.add_argument('--criterion', type=str, default='MSE', choices=['L1', 'MSE'])
    parser.add_argument('--max_edge_len', type=float, default=0.3)
    parser.add_argument('--min_iso_size', type=int, default=100)
    # parser.add_argument('--cascade_list', type=float, nargs='*', default=[0.5, 1.0, 1.5, 2.0])
    # parser.add_argument('--decimate_target', type=int, nargs='*', default=[1e6, 1e6, 1e6, 1e6],
    #                     help="decimate target for number of triangles, <=0 to disable")

    ### mesh options
    # parser.add_argument('--visibility_mask_dilation', type=int, default=10, help="visibility dilation")
    parser.add_argument('--clean_min_f', type=int, default=25,
                        help="mesh clean: min face count for isolated mesh")
    parser.add_argument('--clean_min_d', type=int, default=0,
                        help="mesh clean: min diameter for isolated mesh")
    parser.add_argument('--ssaa', type=int, default=2, help="super sampling anti-aliasing ratio")
    parser.add_argument('--texture_size', type=int, default=4096, help="exported texture resolution")
    parser.add_argument('--refine', action='store_true', help="track face error and do subdivision")
    parser.add_argument("--refine_steps_ratio", type=float, action="append",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    parser.add_argument('--refine_size', type=float, default=0.01, help="refine trig length")
    parser.add_argument('--refine_decimate_ratio', type=float, default=0.1, help="refine decimate ratio")
    parser.add_argument('--refine_remesh_size', type=float, default=0.02, help="remesh trig length")
    parser.add_argument('--pos_gradient_boost', type=float, default=1, help="nvdiffrast option")

    parser.add_argument('--sharp_ratio', type=float, default=0.05, help="sharp ration in normal adjust")
    parser.add_argument('--flat_ratio', type=float, default=0.1, help="flat ration in normal adjust")
    parser.add_argument('--no_mesh_refine', action='store_true', help="do not refine mesh during optimization")
    parser.add_argument('--no_mesh_decimate', action='store_true', help="do not decimate mesh during optimization")

    ### model options
    # parser.add_argument('--backbone', type=str, default='merf_new',
    #                     choices=['merf', 'default', 'linear', 'dense', 'merf_new'], help="backbone type")
    parser.add_argument('--grid_resolution', type=int, default=1024)
    parser.add_argument('--triplane_resolution', type=int, default=-1)

    ### testing options
    parser.add_argument('--save_cnt', type=int, default=1,
                        help="save checkpoints for $ times during training")
    parser.add_argument('--eval_cnt', type=int, default=1,
                        help="perform validation for $ times during training")

    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_no_video', action='store_true', help="test mode: do not save video")
    parser.add_argument('--test_no_baking', action='store_true', help="test mode: do not save baking")
    parser.add_argument('--test_no_mesh', action='store_true', help="test mode: do not save mesh")
    parser.add_argument('--camera_traj', type=str, default='path',
                        help="interp for interpolation, circle for circular camera")

    ### dataset options
    parser.add_argument('--data_format', type=str, default='colmap', choices=['nerf', 'colmap', 'dtu'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'all'])
    parser.add_argument('--test_split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--random_image_batch', action='store_true',
                        help="randomly sample rays from all images per step in training")
    parser.add_argument('--downscale', type=int, default=4, help="downscale training images")
    parser.add_argument('--bound', type=float, default=128,
                        help="assume the scene is bounded in box[-bound, bound]^3, "
                             "if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=-1,
                        help="scale camera location into box[-bound, bound]^3, "
                             "-1 means automatically determine based on camera poses..")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--enable_cam_near_far', action='store_true',
                        help="colmap mode: use the sparse points to estimate camera near far per view.")
    parser.add_argument('--enable_cam_center', action='store_true',
                        help="use camera center instead of sparse point center (colmap dataset only)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--T_thresh', type=float, default=2e-4,
                        help="minimum transmittance to continue ray marching")

    ### optimization options
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")

    ### training options
    parser.add_argument('--iters', type=int, default=20000, help="training iters")
    # parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    # parser.add_argument('--max_steps', type=int, default=1024,
    #                     help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--contract', action='store_true',
                        help="apply spatial contraction as in MERF, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--enable_dense_depth', action='store_true', help="dense depth supervision")
    parser.add_argument('--background', type=str, default='random',
                        choices=['white', 'random', 'last_sample'], help="training background mode")

    # parser.add_argument('--update_extra_interval', type=int, default=16,
    #                     help="iter interval to update extra status (only valid when using --cuda_ray)")
    # parser.add_argument('--max_ray_batch', type=int, default=4096 * 4,
    #                     help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    # parser.add_argument('--grid_size', type=int, default=128, help="density grid resolution")
    parser.add_argument('--mark_untrained', action='store_true', help="mark_untrained grid")
    parser.add_argument('--dt_gamma', type=float, default=1 / 256,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, "
                             ">0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied")
    parser.add_argument('--diffuse_step', type=int, default=0,
                        help="training iters that only trains diffuse color for better initialization")

    # batch size related
    parser.add_argument('--num_rays', type=int, default=4096,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--adaptive_num_rays', action='store_true',
                        help="adaptive num rays for more efficient training")
    parser.add_argument('--num_points', type=int, default=2 ** 18,
                        help="target num points for each training step, only work with adaptive num_rays")

    # regularizations
    parser.add_argument('--lambda_entropy', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_proposal', type=float, default=0,
                        help="loss scale (only for non-cuda-ray mode)")
    parser.add_argument('--lambda_distort', type=float, default=0,
                        help="loss scale (only for non-cuda-ray mode)")
    parser.add_argument('--lambda_specular', type=float, default=1e-5, help="loss scale")
    parser.add_argument('--lambda_depth', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_sparsity', type=float, default=0, help="loss scale")
    # parser.add_argument('--lambda_offsets', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_lap', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_mask', type=float, default=0, help="loss scale")

    ### GUI options
    parser.add_argument('--vis_pose', action='store_true', help="visualize the poses")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1000, help="GUI width")
    parser.add_argument('--H', type=int, default=1000, help="GUI height")
    parser.add_argument('--radius', type=float, default=1, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()

    opt.fp16 = True
    opt.preload = True
    opt.contract = True
    # opt.adaptive_num_rays = True

    # todo add nerf_synthetic
    if opt.data_format == 'nerf':
        # opt.bound = 1
        opt.min_near = 0.05
        opt.contract = False
        opt.decimate_target = [1e5, 1e5, 1e5]
        # opt.random_image_batch = True
        opt.clean_min_f = 16
        opt.clean_min_d = 10
        opt.visibility_mask_dilation = 5
    # todo end

    opt.enable_cam_center = True
    opt.enable_cam_near_far = True

    opt.refine = True
    opt.vert_offset = True
    # opt.no_normal = True

    if opt.vert_offset is False:
        opt.lambda_lap = 0
    opt.max_ray_batch = 2 ** 30  # set to inf to inference the whole image at one render pass
    # inner_idx = (np.array(opt.cascade_list) <= 1.0).sum()
    # opt.cascade_list = opt.cascade_list[inner_idx - 1:]
    # opt.decimate_target = [sum(opt.decimate_target[:inner_idx])] + opt.decimate_target[inner_idx:]

    assert not opt.cuda_ray
    # assert opt.contract
    assert opt.vol_path is not None, f'vol_path is not valid: {opt.vol_path}'

    if opt.contract:
        # mark untrained is not correct in contraction mode...
        opt.mark_untrained = False

    if opt.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif opt.data_format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    else:  # nerf
        from nerf.provider import NeRFDataset

    # convert ratio to steps
    opt.refine_steps = [int(round(x * opt.iters)) for x in opt.refine_steps_ratio]

    seed_everything(opt.seed)

    # if opt.backbone == 'merf_new':
    #     from nerf.network_vosh import NeRFNetwork
    # else:
    #     raise NotImplementedError
    from nerf.network_vosh import NeRFNetwork

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt)

    if opt.criterion == 'L1':
        criterion = torch.nn.SmoothL1Loss(reduction='none')
    elif opt.criterion == 'MSE':
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError

    if opt.test:
        trainer = Trainer('mesh', opt, model, device=device, workspace=opt.workspace, criterion=criterion,
                          fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if not opt.test_no_video:
            test_loader = NeRFDataset(opt, device=device, type=opt.test_split).dataloader()

            if test_loader.has_gt:
                trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]  # set up metrics
                trainer.evaluate(test_loader, name='test')  # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True)  # test and save video

    else:
        lr = opt.lr if 'lr' in opt else 1.0
        optimizer = lambda model: torch.optim.Adam(model.get_params(lr), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        save_interval = max(1, max_epoch // max(1, opt.save_cnt))
        eval_interval = max(1, max_epoch // max(1, opt.eval_cnt))
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}.')

        # colmap can estimate a more compact AABB
        # if not opt.contract and opt.data_format == 'colmap':
        #     model.update_aabb(train_loader._data.pts_aabb)

        # scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer,
        #                                                                 lambda iter: 0.1 ** (iter / opt.iters))
        # scheduler = lambda optimizer: (
        #     torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter:
        #     math.learning_rate_decay(iter + 1, opt.lr_init, opt.lr_final, opt.iters,
        #                              opt.lr_delay_steps, opt.lr_delay_mult)))
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.01 + 0.99 * (iter / 500) if iter <= 500 else
            0.1 ** ((iter - 500) / (opt.iters - 500)))

        trainer = Trainer('mesh', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval,
                          save_interval=save_interval)

        valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()

        trainer.metrics = [PSNRMeter(), ]

        trainer.train(train_loader, valid_loader, max_epoch)

        # last validation
        trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
        trainer.evaluate(valid_loader)

        # also test
        test_loader = NeRFDataset(opt, device=device, type=opt.test_split).dataloader()

        if test_loader.has_gt:
            trainer.evaluate(test_loader, name='test')  # blender has gt, so evaluate it.

        # export refined mesh
        trainer.model.export_mesh_after_refine()

        error_grid_list = sorted(glob.glob(f'{opt.workspace}/error_grid.pt'))
        if not error_grid_list:
            error_grid = trainer.cal_diff(train_loader)

        # mesh_error_grid = trainer.run_mesh_error_epoch(train_loader)
        # trainer.model.remove_faces_in_selected_voxels(error_grid, keep_center=opt.ec_center)

        end_t = time.time()
        print(f"[INFO] mesh stage takes {(end_t - start_t) / 60:.6f} minutes.")
