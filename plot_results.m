function plot_results(rn, foe, fae, t, n, block_str)
    figure('Position',[0, 0, 800, 250])
    grid on
    hold on

    # Plot residuals. Ignore warnings for now since otherwise we'll get some of them due to some deltas being 0.
    #warning('off','all');
    semilogy(n, rn, '1; Rel. residual norm;.-');
    semilogy(n, foe, "markersize", 3, '1; Rel. forward error;o-');
    semilogy(n, fae, '3; Rel. factorization error;.-');
    legend ({
            "Rel. residual norm", 
            "Rel. forward error", 
            "Rel. factorization error"
        }, "location", "eastoutside")
    title (strcat("Error metrics for ", block_str), "fontsize", 16);

    figure('Position',[0, 0, 800, 250])
    grid on
    hold off
    semilogy(n, t, "markersize", 3, '3; Runtime;o-');
    legend ({"Runtime in seconds    "}, "location", "eastoutside");
    ylabel("Runtime in seconds");
    xlabel("n");
    title (strcat("Runtimes for ", block_str), "fontsize", 16);
end