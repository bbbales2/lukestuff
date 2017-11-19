#%%

import numpy

def segment(sample, N):
    im = numpy.array(sample).astype('float')

    minx = float(min(im.flatten()))
    maxx = float(max(im.flatten()))
    im = (im - minx) / (maxx - minx)

    eps = 1.0
    eta = 1e-1
    dt = 1.0
    phi = numpy.zeros(im.shape)

    f = im
    nu = 0.0
    mu = 0.1
    lambda1 = 1.0
    lambda2 = 1.0
    tol = 1e-9
    phi = numpy.zeros(im.shape)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            phi[i, j] = numpy.sin(numpy.pi * i / 5.0) * numpy.sin(numpy.pi * j / 5.0)

    def de(p):
        return eps / (numpy.pi * (p**2 + eps**2))

    for step in range(0, N):
        mask = 1.0 * (phi >= 0)
        mask2 = 1.0 * (phi < 0)
        above = numpy.count_nonzero(mask)
        below = numpy.count_nonzero(mask2)
        c1 = numpy.sum((mask * f).flatten()) / above
        c2 = numpy.sum((mask2 * f).flatten()) / below

        #print above, below, c1, c2

        #if step % N == 0:
        #    plt.subplot(1, 2, 1)
        #    plt.imshow(im, cmap = plt.cm.gray)
            #plt.colorbar(fraction=0.046, pad=0.04)
            #plt.imshow(mahotas.labeled.borders(phi >= 0))
        #    threshold = skimage.filters.threshold_otsu(phi)
        #    mark = mahotas.labeled.borders((phi >= threshold))
        #    plt.imshow(mark)
        #    plt.title('Segmentation', fontsize = 20)
        #    #plt.show()
        #    plt.subplot(1, 2, 2)
        #    plt.imshow(phi, cmap = plt.cm.gray)
        #    #plt.colorbar(fraction=0.046, pad=0.04)
        #    plt.title('Level Set', fontsize = 20)
        #    fig = plt.gcf()
        #    fig.set_size_inches(15,15)
        #    plt.show()

        for i in range(0, phi.shape[0]):
            phi[i, 0] = phi[i, 1]
            phi[i, -1] = phi[i, -2]

        for j in range(0, phi.shape[1]):
            phi[0, j] = phi[1, j]
            phi[-1, j] = phi[-2, j]

        #Approximate derivatives
        phi_x = (phi[1: -1, 2:] - phi[1:-1, :-2]) / 2
        phi_y = (phi[2:, 1: -1] - phi[:-2, 1:-1]) / 2
        phi_xx = (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2])
        phi_yy = (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1])
        phi_xy = (phi[2:, 2:] - phi[:-2, 2:] - phi[2:, :-2] + phi[:-2, :-2]) / 4.0;

        #TV term = Num/Den
        Num = phi_xx * phi_y ** 2 - 2 * phi_x * phi_y * phi_xy + phi_yy * phi_x ** 2;
        Den = numpy.power((phi_x ** 2 + phi_y ** 2), 1.5) + eta;

        #%Compute average colors.
        #%Add to previous iteration of u.
        change = dt * de(phi[1:-1, 1:-1]) * (mu * Num / Den \
                                            - nu \
                                            - lambda1 * (f[1:-1, 1:-1] - c1) ** 2 \
                                            + lambda2 * (f[1:-1, 1:-1] - c2) ** 2)
        norm = numpy.linalg.norm(change.flatten())

        if norm / (phi.shape[0] * phi.shape[1]) < tol:
            print "tolerance reached"
            break

        phi[1:-1, 1:-1] = phi[1:-1, 1:-1] + change

    return phi