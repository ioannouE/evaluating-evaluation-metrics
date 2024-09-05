import argparse
# from . import art_fid
import art_fid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for computing activations.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of threads used for data loading.')
    parser.add_argument('--content_metric', type=str, default='lpips', choices=['lpips', 'vgg', 'alexnet'], help='Content distance.')
    parser.add_argument('--mode', type=str, default='art_fid_inf', choices=['art_fid', 'art_fid_inf'], help='Evaluate ArtFID or ArtFID_infinity.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use.')
    parser.add_argument('--style_images', type=str, required=True, help='Path to style images.')
    parser.add_argument('--content_images', type=str, required=True, help='Path to content images.')
    parser.add_argument('--stylized_images', type=str, required=True, help='Path to stylized images.')
    parser.add_argument('--perturbation', type=float, default=0.1)
    parser.add_argument('--noise', type=int, default=0, help="add noise/perturbation to the image")

    args = parser.parse_args()

    art_fid_value = art_fid.compute_art_fid(args.stylized_images,
                                            args.style_images,
                                            args.content_images,
                                            args.batch_size,
                                            args.device,
                                            args.mode,
                                            args.content_metric,
                                            args.num_workers,
                                            args.perturbation,
                                            args.noise)

    # print('ArtFID value:', art_fid_value)
    print('------------------------------------------------------------------------------------------------------------')
    print('Perturbation: ',  args.perturbation, '   ArtFID value: ', '{:.4f}'.format(art_fid_value))
    print('------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
