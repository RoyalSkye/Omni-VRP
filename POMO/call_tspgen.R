# Rscript call_tspgen.R operator point_lower point_upper ins_num seed

library("netgen")
source("tspgen-master/R/utilities.R")
source("tspgen-master/R/mutator.explosion.R")
source("tspgen-master/R/mutator.implosion.R")
source("tspgen-master/R/mutator.cluster.R")
source("tspgen-master/R/mutator.compression.R")
source("tspgen-master/R/mutator.expansion.R")
source("tspgen-master/R/mutator.grid.R")
source("tspgen-master/R/mutator.linearprojection.R")
source("tspgen-master/R/mutator.rotation.R")

library(ggplot2)
library(gridExtra)

args <- commandArgs(TRUE)
operator <- toString(args[1])
points.num <- as.integer(args[2])
ins.num <- as.integer(args[4])
seed <- as.integer(args[5])
choice <- toString(args[6])
path <- toString(args[7])

#check_internal = ins.num / 5

check_internal = 1

# points.num <- sample(points.lower:points.upper, 1, replace=TRUE)

set.seed(seed)

for (i in 1:ins.num)
{

    x = generateRandomNetwork(n.points = points.num, lower = 0, upper = 1)
    rue = x
    if (operator == "explosion")
    {
        x$coordinates = doExplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "implosion")
    {
        x$coordinates = doImplosionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "cluster")
    {
        x$coordinates = doClusterMutation(x$coordinates, pm=0.4)
    }
    if (operator == "compression")
    {
        x$coordinates = doCompressionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "expansion")
    {
        x$coordinates = doExpansionMutation(x$coordinates, min.eps=0.3, max.eps=0.3)
    }
    if (operator == "grid")
    {
        x$coordinates = doGridMutation(x$coordinates, box.min=0.3, box.max=0.3, p.rot=0, p.jitter=0, jitter.sd=0.05)
    }
    if (operator == "linearprojection")
    {
        x$coordinates = doLinearProjectionMutation(x$coordinates, pm=0.4, p.jitter=0, jitter.sd=0.05)
    }
    if (operator == "rotation")
    {
        x$coordinates = doRotationMutation(x$coordinates, pm=0.4)
    }

    x = rescaleNetwork(x, method = "global2")
    # x$coordinates = x$coordinates * 1000000
    # x$coordinates = round(x$coordinates, 0)
    x$coords = relocateDuplicates(x$coords)
    x$lower = 0
    x$upper = 1
    name = sprintf("data/TSP_test/%s/%d.tsp", operator, i)
    # if (i %% check_internal == 0)
    # {
    #     jpeg(file = paste("data/check/",operator,toString(i), ".jpg", sep = ""), width=1200, height=600)
    #     grid.arrange(autoplot(x), autoplot(rue), nrow = 1L)
    #     dev.off()
    # }

    # print(x$coordinates)
    cat(x$coordinates, "\n", file=path, append=TRUE)

    # exportToTSPlibFormat(x, name, use.extended.format=FALSE)
}
