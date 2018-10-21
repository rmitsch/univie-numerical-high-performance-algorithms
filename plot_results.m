function plot_results(rn, foe, fae, t, n, block_str)
    figure('Position',[0, 0, 800, 250])
    grid on
    hold on

    # Plot residuals. Ignore warnings for now since otherwise we'll get some of them due to some deltas being 0.
    #warning('off','all');
    semilogy(n, rn, "marker", "x",  "markersize", 8, '1; Rel. residual norm;.-');
    semilogy(n, foe, "marker", "+",  "markersize", 8, '2; Rel. forward error;o-');
    semilogy(n, fae, "marker", "s",  "markersize", 8, '3; Rel. factorization error;.-');
    legend ({
            "Rel. residual norm", 
            "Rel. forward error", 
            "Rel. factorization error"
        }, "location", "eastoutside")
    title (strcat("Error metrics for ", block_str), "fontsize", 16);
    ylabel("Metric value");
    xlabel("n");

    figure('Position',[0, 0, 800, 250])
    grid on
    hold off
    semilogy(n, t, "markersize", 3, '3; Runtime;o-');
    legend ({"Runtime in seconds    "}, "location", "eastoutside");
    ylabel("Runtime in seconds");
    xlabel("n");
    title (strcat("Runtimes for ", block_str), "fontsize", 16);
end