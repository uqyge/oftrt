/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    icoFoam

Description
    Transient solver for incompressible, laminar flow of Newtonian fluids.

\*---------------------------------------------------------------------------*/
#include <list>
#include "fvCFD.H"
#include "pisoControl.H"

#include "fpe.H"
// #include "sampleUffMNIST.H"
#include "uffModel.H"
samplesCommon::Args args;
MNISTSampleParams params = initializeSampleParams(args);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
#include "setRootCase.H"
#include "createTime.H"
#include "createMesh.H"

    pisoControl piso(mesh);

#include "createFields.H"
#include "initContinuityErrs.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info << "\nStarting time loop\n"
         << endl;

    // initial model object
    params.dataDirs.push_back("./data/");
    params.batchSize = 1024;
    params.uffFileName = "test.uff";

    uffModel sample(params);

    FPExceptionsGuard fpguard;
    sample.build();

    while (runTime.loop())
    {
        Info << "Time = " << runTime.timeName() << nl << endl;

        std::vector<float> output_real;

        const int scaling = 1;
        static std::vector<float> input_f_pv_zeta;
        input_f_pv_zeta.clear();
        int i = 0;
        const cellList &cells = mesh.cells();
        forAll(cells, celli)
        {
            for (int j = 0; j < scaling; j++)
            {
                input_f_pv_zeta.emplace_back(in_1[celli]);
                input_f_pv_zeta.emplace_back(in_3[celli]);
                input_f_pv_zeta.emplace_back(in_2[celli]);
            }
        }
        std::cout << "input size: " << input_f_pv_zeta.size() << '\n';
        auto t_start = std::chrono::high_resolution_clock::now();
        bool success = sample.infer(input_f_pv_zeta, output_real);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        std::cout << "inference takes " << total << "ms.\n";
        std::cout << success
                  << " uff inference finished.\n";
        std::cout << "output size " << output_real.size() << "\n";

        float min_max[22] = {
            1.0000000000000000,
            0.0000000000000000,
            0.2330000000000000,
            0.0000000000000000,
            0.1300660000000000,
            0.0000000000000000,
            0.1029540000000000,
            0.0000000000000000,
            0.1371060000000000,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000,
            2230.4499999999998181,
            283.9370000000000118,
            21531.2999999999992724,
            -0.0000000180828000,
            0.0001034380000000,
            0.0000253753000000,
            0.0000733570000000,
            0.0000175979000000,
            0.0000115664000000,
            0.0000014680900000};
        auto t_w_start = std::chrono::high_resolution_clock::now();

        std::vector<volScalarField> out_fields = {
            out_1, out_2, out_3, out_4,
            out_5, out_6, out_7, out_8,
            out_9, out_10, out_11};

        forAll(cells, celli)
        {
            int idx = 0;
            for (volScalarField &out_f : out_fields)
            {
                // std::cout << output_real[celli * 11 + idx] << '\n';
                out_f[celli] = output_real[celli * 11 + idx] * (min_max[2 * idx] - min_max[2 * idx + 1]) + min_max[2 * idx + 1];
                // std::cout << out_f[celli] << '\n';
                idx++;
            }
        }
        out_1 = out_fields[0];
        out_2 = out_fields[1];
        out_3 = out_fields[2];
        out_4 = out_fields[3];
        out_5 = out_fields[4];
        out_6 = out_fields[5];
        out_7 = out_fields[6];
        out_8 = out_fields[7];
        out_9 = out_fields[8];
        out_10 = out_fields[9];
        out_11 = out_fields[10];

        auto t_w_end = std::chrono::high_resolution_clock::now();
        auto total_w = std::chrono::duration<float, std::milli>(t_w_end - t_w_start).count();
        // forAll(cells, celli)
        // {
        //     std::cout << out_7[celli] << '\n';
        // }
        std::cout << "write takes " << total_w << "ms.\n";

#include "CourantNo.H"

        // Momentum predictor

        fvVectorMatrix UEqn(
            fvm::ddt(U) + fvm::div(phi, U) - fvm::laplacian(nu, U));

        if (piso.momentumPredictor())
        {
            solve(UEqn == -fvc::grad(p));
        }

        // --- PISO loop
        while (piso.correct())
        {
            volScalarField rAU(1.0 / UEqn.A());
            volVectorField HbyA(constrainHbyA(rAU * UEqn.H(), U, p));
            surfaceScalarField phiHbyA(
                "phiHbyA",
                fvc::flux(HbyA) + fvc::interpolate(rAU) * fvc::ddtCorr(U, phi));

            adjustPhi(phiHbyA, U, p);

            // Update the pressure BCs to ensure flux consistency
            constrainPressure(p, U, phiHbyA, rAU);

            // Non-orthogonal pressure corrector loop
            while (piso.correctNonOrthogonal())
            {
                // Pressure corrector

                fvScalarMatrix pEqn(
                    fvm::laplacian(rAU, p) == fvc::div(phiHbyA));

                pEqn.setReference(pRefCell, pRefValue);

                pEqn.solve(mesh.solver(p.select(piso.finalInnerIter())));

                if (piso.finalNonOrthogonalIter())
                {
                    phi = phiHbyA - pEqn.flux();
                }
            }

#include "continuityErrs.H"

            U = HbyA - rAU * fvc::grad(p);
            U.correctBoundaryConditions();
        }

        runTime.write();

        Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = " << runTime.elapsedClockTime() << " s"
             << nl << endl;
    }
    sample.teardown();
    Info << "End\n"
         << endl;

    return 0;
}

// ************************************************************************* //
