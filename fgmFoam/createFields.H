Info << "Reading transportProperties\n"
     << endl;

IOdictionary transportProperties(
    IOobject(
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE));

dimensionedScalar nu(
    "nu",
    dimViscosity,
    transportProperties.lookup("nu"));

Info << "Reading field p\n"
     << endl;
volScalarField p(
    IOobject(
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading field input_1\n"
     << endl;
volScalarField in_1(
    IOobject(
        "in_1",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);
Info << "Reading field input_2\n"
     << endl;
volScalarField in_2(
    IOobject(
        "in_2",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading field input_3\n"
     << endl;
volScalarField in_3(
    IOobject(
        "in_3",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_1(
    IOobject(
        "out_1",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_2(
    IOobject(
        "out_2",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_3(
    IOobject(
        "out_3",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_4(
    IOobject(
        "out_4",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_5(
    IOobject(
        "out_5",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_6(
    IOobject(
        "out_6",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_7(
    IOobject(
        "out_7",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_8(
    IOobject(
        "out_8",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_9(
    IOobject(
        "out_9",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_10(
    IOobject(
        "out_10",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

volScalarField out_11(
    IOobject(
        "out_11",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading field U\n"
     << endl;
volVectorField U(
    IOobject(
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

#include "createPhi.H"

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, mesh.solutionDict().subDict("PISO"), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());
