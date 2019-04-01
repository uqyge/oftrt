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
template <class T>
T cubic(T x)
{
    return x * x * x;
}
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
    params.batchSize = 1024 * 4;
    params.uffFileName = "test.uff";
    params.inputTensorNames.push_back("input_1");
    params.outputTensorNames.push_back("dense_2/BiasAdd");

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
        // int i = 0;
        float in_scaler[6] = {
            0.4999999999999999,
            0.2892519086652557,
            0.4949999999959955,
            0.3159509455594174,
            0.5000000000000324,
            0.2892519086655237};
        const cellList &cells = mesh.cells();
        forAll(cells, celli)
        {
            for (int j = 0; j < scaling; j++)
            {
                input_f_pv_zeta.emplace_back((in_1[celli] - in_scaler[0]) / in_scaler[1]);
                input_f_pv_zeta.emplace_back((in_2[celli] - in_scaler[2]) / in_scaler[3]);
                input_f_pv_zeta.emplace_back((in_3[celli] - in_scaler[4]) / in_scaler[5]);
            }
        }
        std::cout << "input size: " << input_f_pv_zeta.size() << '\n';
        auto t_start = std::chrono::high_resolution_clock::now();
        // bool success = sample.infer(input_f_pv_zeta, output_real);
        // bool success = sample.infer(input_f_pv_zeta, output_real);
        // success = sample.infer(input_f_pv_zeta, output_real);
        bool success = sample.infer_new(input_f_pv_zeta, output_real);
        for (int i = 0; i < 100; i++)
        {
            success = sample.infer_new(input_f_pv_zeta, output_real);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        auto total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        std::cout << "inference takes " << total << "ms.\n";
        std::cout << success
                  << " uff inference finished.\n";
        std::cout << "output size " << output_real.size() << "\n";

        float min_max[30] = {
            0.1121079534440103,
            0.0426125035204322,
            0.0050576506362650,
            0.0060149066951871,
            0.0074201615268215,
            0.0102671571976970,
            0.3902602471463928,
            0.1363730691662354,
            0.0110028139562614,
            0.0147071294580590,
            0.2513777876969300,
            0.1073862973086250,
            0.0167120604981467,
            0.0093845131656524,
            0.0182667525221380,
            0.0159984620004068,
            0.7350347640886691,
            0.2062934637594329,
            0.2152808969457236,
            0.1000664830825949,
            0.1627449117417809,
            0.0802050096256412,
            0.0570861219763779,
            0.0273965641902137,
            0.6865258073725889,
            0.1769577281142915,
            8.0553410390390656,
            1.1661353060101767,
            2.7951803349889142,
            3.1225226020677006};
        auto t_w_start = std::chrono::high_resolution_clock::now();

        std::vector<volScalarField> out_fields(15, in_1);

        forAll(cells, celli)
        {
            int idx = 0;
            for (volScalarField &out_f : out_fields)
            {
                // std::cout << output_real[celli * 11 + idx] << '\n';
                out_f[celli] = max(cubic((output_real[celli * 15 + idx] * min_max[2 * idx + 1]) + min_max[2 * idx]), 0);
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
        out_12 = out_fields[11];
        out_13 = out_fields[12];
        out_14 = out_fields[13];
        out_15 = out_fields[14];

        auto t_w_end = std::chrono::high_resolution_clock::now();
        auto total_w = std::chrono::duration<float, std::milli>(t_w_end - t_w_start).count();
        // forAll(cells, celli)
        // {
        //     std::cout << out_7[celli] << '\n';
        // }
        std::cout << "write takes " << total_w << "ms.\n";

        // #include "CourantNo.H"

        // Momentum predictor

        //         fvVectorMatrix UEqn(
        //             fvm::ddt(U) + fvm::div(phi, U) - fvm::laplacian(nu, U));

        //         if (piso.momentumPredictor())
        //         {
        //             solve(UEqn == -fvc::grad(p));
        //         }

        //         // --- PISO loop
        //         while (piso.correct())
        //         {
        //             volScalarField rAU(1.0 / UEqn.A());
        //             volVectorField HbyA(constrainHbyA(rAU * UEqn.H(), U, p));
        //             surfaceScalarField phiHbyA(
        //                 "phiHbyA",
        //                 fvc::flux(HbyA) + fvc::interpolate(rAU) * fvc::ddtCorr(U, phi));

        //             adjustPhi(phiHbyA, U, p);

        //             // Update the pressure BCs to ensure flux consistency
        //             constrainPressure(p, U, phiHbyA, rAU);

        //             // Non-orthogonal pressure corrector loop
        //             while (piso.correctNonOrthogonal())
        //             {
        //                 // Pressure corrector

        //                 fvScalarMatrix pEqn(
        //                     fvm::laplacian(rAU, p) == fvc::div(phiHbyA));

        //                 pEqn.setReference(pRefCell, pRefValue);

        //                 pEqn.solve(mesh.solver(p.select(piso.finalInnerIter())));

        //                 if (piso.finalNonOrthogonalIter())
        //                 {
        //                     phi = phiHbyA - pEqn.flux();
        //                 }
        //             }

        // #include "continuityErrs.H"

        //             U = HbyA - rAU * fvc::grad(p);
        //             U.correctBoundaryConditions();
        //         }

        runTime.write();

        //         Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        //              << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        //              << nl << endl;
    }
    sample.teardown();
    Info << "End\n"
         << endl;

    return 0;
}

// ************************************************************************* //
