using HDF5
using StaticArrays
using Statistics
using Random
using LinearAlgebra
#using SpecialFunctions
using Printf
using Parameters
using .Threads

using PyPlot

const SOLAMASS = 1.989e+43
const PROTONMASS = 1.6726e-24
const XH = 0.71
const BOLTZMANN = 1.38e-16
const GAMMA = 5.0 / 3

const UnitMass_in_g = 1.989e+43
const UnitLength_in_cm = 3.08568e+21
const UnitVelocity_in_cm_per_s = 1e5

@with_kw struct Params{NDIM,T}

    LX::T
    LZ::T
    scale_height::T
    T_mu::T

    Mgas_tot::T
    mgas::T = 0.0
    Ngas::Int64 = (mgas > 0.) ? round(Int64, Mgas_tot / mgas) : Ngas

    u0::T = T_mu * BOLTZMANN / (GAMMA-1.0) / PROTONMASS / UnitVelocity_in_cm_per_s^2

    seed::Int64 = 0

    filename::String = ""
end

function generate_ICfile_glass_box(par::Params{NDIM,T}) where{NDIM,T}

    @unpack LX, LZ, T_mu, Ngas, mgas, Mgas_tot, u0, scale_height, filename, seed = par

    mgas = Mgas_tot / Ngas

    #@show mgas, Ngas, (Mgas_tot/LX^2)
    println("Ngas = ", Ngas, "  mgas = ", mgas, "  surface density = ", Mgas_tot/LX^2)
    id_gas = collect(Int32, 1:Ngas)
    mass = ones(Ngas) .* mgas
    u_gas = ones(Ngas) .* u0;
    #hsml = ones(Ngas) .* h0

    ########## prepare glass configuration ##########
    pos = zeros(NDIM, Ngas);
    vel = zeros(NDIM, Ngas);

    Random.seed!(seed);

    z0 = 0.5 * LZ
    #z0 = 0
    z_prop = 1e3
    for i in 1:Ngas
        for j in 1:NDIM-1
            pos[j,i] = rand() * LX
        end
        while true
            r = rand()
            z_prop = scale_height * ( 0.5 * log( (1. + r)/(1. - r) ) )
            if z_prop < 0.5 * LZ
                break
            end
        end
        fac = (i % 2) * 2 - 1 #+1 or -1
        pos[NDIM,i] = fac * z_prop
    end


    if filename == ""
        filename = "ics"
        filename *= "_Mtot" * @sprintf("%.0e", Mgas_tot) *
                    "_u" * @sprintf("%.0e", u0) *
                    "_LX" * @sprintf("%.0e", LX) *
                    "_LZ" * @sprintf("%.0e", LZ) *
                    "_SH" * @sprintf("%.0e", scale_height) *
                    "_Ngas" * @sprintf("%.0e", Ngas) *
                    "_mgas" * @sprintf("%.1e", mgas) * ".hdf5"
    end

    ########## write to file ##########
    println("saving to file...")
    save_gadget_ics(filename, pos, vel, id_gas, mass, u_gas, Ngas, LX)
    println("done")

    return pos, vel, id_gas, mass, u_gas, Ngas, LX
end

function save_gadget_ics(filename, pos, vel, id_gas, mass, u_gas, Ngas, boxsize)
    T = Float32

    fid=h5open(filename,"w")

    grp_head = g_create(fid,"Header");
    attrs(fid["Header"])["NumPart_ThisFile"]       = Int32[Ngas, 0, 0, 0, 0, 0]
    attrs(fid["Header"])["MassTable"]              = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    attrs(fid["Header"])["Time"]                   = 0.0
    attrs(fid["Header"])["Redshift"]               = 0.0
    attrs(fid["Header"])["Flag_Sfr"]               = 0
    attrs(fid["Header"])["Flag_Feedback"]          = 0
    attrs(fid["Header"])["NumPart_Total"]          = Int32[Ngas, 0, 0, 0, 0, 0]
    attrs(fid["Header"])["Flag_Cooling"]           = 0
    attrs(fid["Header"])["NumFilesPerSnapshot"]    = 1
    attrs(fid["Header"])["BoxSize"]                = boxsize
    attrs(fid["Header"])["Omega0"]                 = 0.27
    attrs(fid["Header"])["OmegaLambda"]            = 0.73
    attrs(fid["Header"])["HubbleParam"]            = 1.0
    attrs(fid["Header"])["Flag_StellarAge"]        = 0
    attrs(fid["Header"])["Flag_Metals"]            = 0
    attrs(fid["Header"])["NumPart_Total_HighWord"] = UInt32[0,0,0,0,0,0]
    attrs(fid["Header"])["flag_entropy_instead_u"] = 0
    attrs(fid["Header"])["Flag_DoublePrecision"]   = 0
    attrs(fid["Header"])["Flag_IC_Info"]           = 0
    #attrs(fid["Header"])["lpt_scalingfactor"] =

    grp_part = g_create(fid,"PartType0");
    h5write(filename, "PartType0/Coordinates"   , T.(pos))
    h5write(filename, "PartType0/Velocities"    , T.(vel))
    h5write(filename, "PartType0/ParticleIDs"   , id_gas)
    h5write(filename, "PartType0/Masses"        , T.(mass))
    h5write(filename, "PartType0/InternalEnergy", T.(u_gas))

    close(fid)
end

par = Params{3,Float64}(T_mu=1e4, LX=1., LZ=10, Ngas=10000000, Mgas_tot=1e-3, scale_height=0.25);
pos, vel, id_gas, mass, u, Ngas, LX = generate_ICfile_glass_box(par);
clf()
plot(pos[1,:], pos[3,:], ".", ms=0.3)
