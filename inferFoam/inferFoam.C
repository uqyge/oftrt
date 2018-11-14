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

#include "fvCFD.H"
#include "pisoControl.H"

#include "fpe.H"
#include "sampleUffMNIST.H"
// #include "sampleMNIST.H"
// samplesCommon::Args args;
// MNISTSampleParams params = initializeSampleParams(args);
// std::string locateFile(const std::string &input);
// using namespace Foam;
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
    // bool success = sample.infer(b);
    while (runTime.loop())
    {

        Info << "Time = " << runTime.timeName() << nl << endl;
        // bool success = sample.infer(b);
        float inputs[in_1.size() * 2];
        // std::vector<float> inputs[in_1.size() * 2];
        const cellList &cells = mesh.cells();
        int i = 0;
        forAll(cells, celli)
        {
            inputs[i++] = in_1[celli];
            inputs[i++] = in_2[celli];
            // Info << in_1[celli] << endl;
        }
        // Info << input.size() << endl;
        Info << "inputs:" << inputs[0] << '\n';
        Info << "inputs:" << inputs[1] << '\n';
        std::vector<float> input_p_he(inputs, inputs + in_1.size() * 2);
        Info << "input_p_he:" << input_p_he[0] << '\n';
        Info << "input_p_he:" << input_p_he[1] << '\n';

        // // uff loader
        int maxBatchSize = 10;
        int batchSize = 2;

        std::vector<float> output_real;

        auto parser = createUffParser();
        parser->registerInput("input_1", Dims3(1, 2, 1), UffInputOrder::kNCHW);
        // parser->registerInput("input_1", Dims2(2, 1), UffInputOrder::kNCHW);
        parser->registerOutput("dense_2/BiasAdd");

        auto modelFile = locateFile("mayer.uff");
        std::cout << "uff:" << modelFile << '\n';
        std::cout << parser << '\n';

        FPExceptionsGuard fpguard;
        nvinfer1::ICudaEngine *engine = loadModelAndCreateEngine(modelFile.c_str(), maxBatchSize, parser);
        if (!engine)
            std::cout << "engine fail\n";

        parser->destroy();

        execute(*engine, batchSize, input_p_he, output_real);
        engine->destroy();
        forAll(cells, celli)
        {
            out_1[celli] = output_real[celli * 4];
            out_2[celli] = output_real[celli * 4 + 1];
            out_3[celli] = output_real[celli * 4 + 2];
            out_4[celli] = output_real[celli * 4 + 3];
        }

        // for (int i = 0; i < output_real.size(); i++)
        // {
        //     std::cout << "output_real " << i << ":" << output_real[i] << '\n';
        // }
        std::cout << "uff inference finished.\n";

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

    Info << "End\n"
         << endl;

    return 0;
}

// ************************************************************************* //
