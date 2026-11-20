"""Clear external character publication grants during offline recovery."""

from secrets import randbits


def reset_restored_sharing(*, has_publication_id=True):
    # Rotate the epoch for every role, including roles with only a pending preview.
    # A restored counter cannot simply advance by one: newer previews may use it.
    publication = ",publication_id=NULL" if has_publication_id else ""
    return (
        "UPDATE character_library SET share_token=NULL,shared_revision=NULL,sharing_epoch=:epoch" + publication,
        {"epoch": (1 << 62) | randbits(62)},
    )
